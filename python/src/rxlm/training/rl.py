import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import TypedDict, Optional, Literal
from .utils import TokenizedDict
from .ddp import distributed_mean


class RlAlgorithm(ABC):
    def __init__(self):
        super(RlAlgorithm, self).__init__()
        self.critic_loss_fn = nn.MSELoss()

    @abstractmethod
    def policy_loss(self, query: TokenizedDict, answer: TokenizedDict, logits: torch.Tensor,
                    old_log_probs: torch.Tensor, advantages: torch.Tensor, prev_stm_state: torch.Tensor, new_stm_state: torch.Tensor,
                    ref_log_probs: torch.Tensor, step: int, prev_step_log_probs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        pass

    @abstractmethod
    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def critic_loss(self, values: torch.Tensor, ref_values: torch.Tensor) -> torch.Tensor:
        return self.critic_loss_fn(values, ref_values)


class PPOConfig(TypedDict):
    clip_eps: Optional[float]
    gae_lambda: Optional[float]
    gae_gamma: Optional[float]
    entropy_coef: Optional[float]
    use_distributed_advantage_norm: Optional[bool]
    clip_critic_values: Optional[bool]
    critic_value_clip: Optional[float]
    debug_mode: Optional[bool]
    debug_interval: Optional[int]
    ref_kl_coef: Optional[float]


class PPOAlgorithm(RlAlgorithm):
    """
    Proximal Policy Optimization (PPO) algorithm for MRL

    Note: in first MRL experiments using GAE advantages for step caused incorrect policy updates,
    so it's recommended to use modified Implicit Memory Policy Optimization (IMPO) algorithm, with
    simplified advantages and additional loss terms for memory regularization.
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        super(PPOAlgorithm, self).__init__()

        if config is None:
            config = {}

        # PPO Config
        self.clip_eps = config.get('clip_eps', 0.2)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.gae_gamma = config.get('gae_gamma', 0.99)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.ref_kl_coef = config.get('ref_kl_coef', 0.05)
        self.use_distributed_advantage_norm = config.get('use_distributed_advantage_norm', False)
        self.clip_critic_values = config.get('clip_critic_values', True)
        self.critic_value_clip = config.get('critic_value_clip', 20.0)
        self.debug_mode = config.get('debug_mode', False)
        self.debug_interval = config.get('debug_interval', 10)
        self.debug_step = 0

    def critic_loss(self, values: torch.Tensor, ref_values: torch.Tensor) -> torch.Tensor:
        # Critic loss with clipped values
        if self.clip_critic_values:
            values = torch.clamp(values, -self.critic_value_clip, self.critic_value_clip)
            ref_values = torch.clamp(ref_values, -self.critic_value_clip, self.critic_value_clip)
        return self.critic_loss_fn(values, ref_values)

    def policy_loss(self, query: TokenizedDict, answer: TokenizedDict, logits: torch.Tensor,
                    old_log_probs: torch.Tensor, advantages: torch.Tensor, prev_stm_state: torch.Tensor, new_stm_state: torch.Tensor,
                    ref_log_probs: torch.Tensor, step: int, prev_step_log_probs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # 1. Get query, answer, max and combined lengths in batch
        query_lens = query['attention_mask'].sum(dim=1).long()  # Query lengths per sample
        answer_mask = answer['attention_mask']
        answer_lens = answer_mask.sum(dim=1).long()  # Answer lengths per sample (before padding)
        max_length = query['input_ids'].size(1)

        combined_lens = torch.minimum(
            query_lens + answer_lens,
            torch.full_like(query_lens, max_length)
        )
        # 2. Extract only answer logits
        batch_size, _, vocab_size = logits.size()
        new_logits = torch.zeros((batch_size, max_length, vocab_size), dtype=logits.dtype, device=logits.device)

        for i in range(batch_size):
            start = query_lens[i].item()
            end = combined_lens[i].item()
            valid_len = end - start
            if valid_len > 0:
                new_logits[i, :valid_len] = logits[i, start:end]

        # 3. Shift sequences for correct probabilities alignment
        shifted_logits = new_logits[:, :-1, :] # Remove last sequence element logits - most likely padding or [EOS]
        shifted_targets = answer['input_ids'][:, 1:] # Remove first answer token - deterministic [A] token
        shifted_mask = answer_mask[:, 1:] # Remove also first position from attention mask
        shifted_old_log_probs = old_log_probs[:, 1:] # And from old log probs - it's for [A] deterministic token

        # 4. Calculate and mask new shifted log probs
        new_log_probs = F.log_softmax(shifted_logits, dim=-1)
        shifted_log_probs = new_log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
        shifted_log_probs *= shifted_mask
        shifted_old_log_probs *= shifted_mask

        # 5. Calculate ratio
        ratio = (shifted_log_probs - shifted_old_log_probs).exp()

        advantages = advantages.unsqueeze(-1)

        # 6. Log most important stats in debug mode
        if self.debug_mode:
            if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
                self.debug_step = 1
                print(
                    f"Logits stats: min={new_logits.min().item():.4f}, max={new_logits.max().item():.4f}, mean={new_logits.mean().item():.4f}")
                print(
                    f"Ratio stats: min={ratio.min().item():.4f}, max={ratio.max().item():.4f}, mean={((ratio * shifted_mask).sum(dim=-1) / (shifted_mask.sum(dim=-1) + 1e-8)).mean().item():.4f}")
                print(
                    f"Advantage stats: min={advantages.min().item():.4f}, max={advantages.max().item():.4f}, mean={advantages.mean().item():.4f}")
            else:
                self.debug_step += 1

        # 7. Calculate base policy loss
        surr1 = (ratio * shifted_mask) * advantages
        surr2 = (torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * shifted_mask) * advantages
        policy_loss = -(torch.min(surr1, surr2).sum(dim=-1) / (shifted_mask.sum(dim=-1) + 1e-8)).mean()

        # 8. Add Entropy bonus
        entropy_mask = answer_mask[:, :-1]
        entropy = -(
            (new_log_probs * new_log_probs.exp() * entropy_mask.unsqueeze(-1)).sum(dim=-1) / (entropy_mask.sum(dim=-1).unsqueeze(-1) + 1e-8)
        ).mean()
        policy_loss -= self.entropy_coef * entropy

        # 9. Reference KL-div penalty
        kl_loss = self.kl_penalty(ref_log_probs, shifted_log_probs, shifted_mask)
        if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
            print(
                f'---- KL penalty: {kl_loss.item():.4f}, scaled: {(self.ref_kl_coef * kl_loss).item():.4f}')
        policy_loss += self.ref_kl_coef * kl_loss

        return policy_loss, new_log_probs.clone().detach(), {}

    def kl_penalty(self, ref_log_probs: torch.Tensor, new_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kl = (ref_log_probs.exp() * (ref_log_probs - new_log_probs)) * mask
        kl_loss = kl.sum() / (mask.sum() + 1e-8)
        return kl_loss

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     last_value: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_len, batch_size = rewards.shape
        advantages = torch.zeros_like(rewards, device=rewards.device)
        last_advantage = 0
        next_value = last_value
        dones = dones.float()

        for t in reversed(range(trajectory_len)):
            # Calculate delta from rewards, stored next_value, masked by stored next_done, and values
            delta = rewards[t] + self.gae_gamma * next_value * (1 - dones[t]) - values[t]
            # Calculate advantages based on delta, gamma/lambda factors and last advantage, masked by current done flags
            advantages[t] = delta + self.gae_gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            # Store current step data as last_advantage, next_done and next_value, for the next iteration step
            last_advantage = advantages[t]
            next_value = values[t]

        # Calculate reference returns, based on advantages and values, and return them with advantages for critic update
        returns = advantages + values
        return advantages, returns

    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        advantages, ref_values = self._compute_gae(rewards[:-1], values[:-1], values[-1], dones[:-1])

        if self.use_distributed_advantage_norm:
            mean_advantage = distributed_mean(advantages.mean())
            std_advantage = distributed_mean(advantages.std())
            normalized_advantages = (advantages - mean_advantage) / (std_advantage + 1e-8)
        else:
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return normalized_advantages, ref_values


class IMPOConfig(TypedDict):
    clip_eps: Optional[float]
    use_gae: Optional[bool]
    gae_lambda: Optional[float]
    gae_gamma: Optional[float]
    entropy_coef: Optional[float]
    use_distributed_advantage_norm: Optional[bool]
    clip_critic_values: Optional[bool]
    critic_value_clip: Optional[float]
    debug_mode: Optional[bool]
    debug_interval: Optional[int]
    ref_kl_coef: Optional[float]
    interstep_kl_coef: Optional[float]
    stm_diff_coef: Optional[float]
    use_stm_cosine_sim: Optional[bool]
    cosine_sim_coef: Optional[float]
    advantage_mode: Literal['batch', 'step', 'all']


class IMPOAlgorithm(RlAlgorithm):
    """
    Implicit Memory Policy Optimization (IMPO) algorithm for Memory Reinforcement Learning.

    It's a modified version of PPO with simplified advantages and additional loss terms:
    - STM diff loss (MSE) - with coef based on square root of current step number (each next
      step should have smaller STM update) - `sqrt(step + 1) * stm_diff_coef * mse(new_stm, old_stm)`
    - Policy Consistency Loss - KL div between current and previous step policies (interactions with same
      topic should have similar policies)

    Algorithm results in constant reward improvement in MRL training from the first steps
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        super(IMPOAlgorithm, self).__init__()

        if config is None:
            config = {}

        # PPO Config
        self.clip_eps = config.get('clip_eps', 0.2)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.gae_gamma = config.get('gae_gamma', 0.99)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.use_distributed_advantage_norm = config.get('use_distributed_advantage_norm', False)
        self.clip_critic_values = config.get('clip_critic_values', True)
        self.critic_value_clip = config.get('critic_value_clip', 20.0)
        self.debug_mode = config.get('debug_mode', False)
        self.debug_interval = config.get('debug_interval', 10)
        self.use_gae = config.get('use_gae', False)
        self.ref_kl_coef = config.get('ref_kl_coef', 0.05)
        self.interstep_kl_coef = config.get('interstep_kl_coef', 0.01)
        self.stm_diff_coef = config.get('stm_diff_coef', 0.0001) # should be higher for non-sigmoid residual gates
        self.use_stm_cosine_sim = config.get('use_stm_cosine_sim', False)
        self.cosine_sim_coef = config.get('cosine_sim_coef', 0.01)
        self.debug_step = 0
        self.advantage_mode = config.get('advantage_mode', 'all')

        # Additional losses
        self.policy_consistency_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.stm_diff_loss_fn = nn.MSELoss()

    def critic_loss(self, values: torch.Tensor, ref_values: torch.Tensor) -> torch.Tensor:
        # Critic loss with clipped values
        if self.clip_critic_values:
            values = torch.clamp(values, -self.critic_value_clip, self.critic_value_clip)
            ref_values = torch.clamp(ref_values, -self.critic_value_clip, self.critic_value_clip)
        return self.critic_loss_fn(values, ref_values)

    def policy_loss(self, query: TokenizedDict, answer: TokenizedDict, logits: torch.Tensor,
                    old_log_probs: torch.Tensor, advantages: torch.Tensor, prev_stm_state: torch.Tensor, new_stm_state: torch.Tensor,
                    ref_log_probs: torch.Tensor, step: int, prev_step_log_probs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # 1. Get query, answer, max and combined lengths in batch
        query_lens = query['attention_mask'].sum(dim=1).long()  # Query lengths per sample
        answer_mask = answer['attention_mask']
        answer_lens = answer_mask.sum(dim=1).long()  # Answer lengths per sample (before padding)
        max_length = query['input_ids'].size(1)

        combined_lens = torch.minimum(
            query_lens + answer_lens,
            torch.full_like(query_lens, max_length)
        )
        # 2. Extract only answer logits
        batch_size, _, vocab_size = logits.size()
        new_logits = torch.zeros((batch_size, max_length, vocab_size), dtype=logits.dtype, device=logits.device)

        for i in range(batch_size):
            start = query_lens[i].item()
            end = combined_lens[i].item()
            valid_len = end - start
            if valid_len > 0:
                new_logits[i, :valid_len] = logits[i, start:end]

        # 3. Shift sequences for correct probabilities alignment
        shifted_logits = new_logits[:, :-1, :] # Remove last sequence element logits - most likely padding or [EOS]
        shifted_targets = answer['input_ids'][:, 1:] # Remove first answer token - deterministic [A] token
        shifted_mask = answer_mask[:, 1:] # Remove also first position from attention mask

        # 4. Calculate and mask new shifted log probs
        new_log_probs = F.log_softmax(shifted_logits, dim=-1)
        shifted_log_probs = new_log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
        shifted_log_probs *= shifted_mask
        old_log_probs *= shifted_mask
        ref_log_probs *= shifted_mask

        # 5. Calculate ratio
        ratio = (shifted_log_probs - old_log_probs).exp()

        advantages = advantages.unsqueeze(-1)

        # 6. Log most important stats in debug mode
        if self.debug_mode:
            if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
                self.debug_step = 1
                print(
                    f"-- Logits stats: min={new_logits.min().item():.4f}, max={new_logits.max().item():.4f}, mean={new_logits.mean().item():.4f}")
                print(
                    f"-- Ratio stats: min={ratio.min().item():.4f}, max={ratio.max().item():.4f}, mean={((ratio * shifted_mask).sum(dim=-1) / (shifted_mask.sum(dim=-1) + 1e-8)).mean().item():.4f}")
                print(
                    f"-- Advantage stats: min={advantages.min().item():.4f}, max={advantages.max().item():.4f}, mean={advantages.mean().item():.4f}")
            else:
                self.debug_step += 1

        # 7. Calculate base policy loss
        surr1 = (ratio * shifted_mask) * advantages
        surr2 = (torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * shifted_mask) * advantages
        policy_loss = -(torch.min(surr1, surr2).sum(dim=-1) / (shifted_mask.sum(dim=-1) + 1e-8)).mean()

        # 8. Add Entropy bonus
        entropy_mask = answer_mask[:, :-1]
        entropy = -(
            (new_log_probs * new_log_probs.exp() * entropy_mask.unsqueeze(-1)).sum(dim=-1) / (entropy_mask.sum(dim=-1).unsqueeze(-1) + 1e-8)
        ).mean()
        if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
            print(
                f'---- Entropy bonus: {entropy.item():.4f}, scaled: {(self.entropy_coef * entropy).item():.4f}')
        policy_loss -= self.entropy_coef * entropy

        # 9. Reference KL-div penalty
        kl_loss = self.kl_penalty(ref_log_probs, shifted_log_probs, shifted_mask)
        if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
            print(
                f'---- KL penalty: {kl_loss.item():.4f}, scaled: {(self.ref_kl_coef * kl_loss).item():.4f}')
        policy_loss += self.ref_kl_coef * kl_loss

        # 10. Calculate step policy consistency loss
        if prev_step_log_probs is not None:
            interstep_kl_loss = self.policy_consistency_loss(prev_step_log_probs, new_log_probs.exp(), entropy_mask)
            if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
                print(f'---- Interstep KL loss: {interstep_kl_loss.item():.4f}, scaled: {(self.interstep_kl_coef * interstep_kl_loss).item():.4f}')
            policy_loss += self.interstep_kl_coef * interstep_kl_loss
        else:
            interstep_kl_loss = None

        # 11. Calculate STM diff loss
        mem_diff_scale = torch.sqrt(torch.tensor(step + 1).to(new_stm_state.device)) * self.stm_diff_coef
        if self.use_stm_cosine_sim:
            mem_sim = F.cosine_similarity(new_stm_state, prev_stm_state, dim=-1).mean()
            policy_loss -= self.cosine_sim_coef * mem_sim
        else:
            mem_sim = torch.tensor(0.0)

        mem_diff_loss = self.stm_diff_loss_fn(new_stm_state, prev_stm_state)
        policy_loss += mem_diff_scale * mem_diff_loss

        return policy_loss, new_log_probs.clone().detach(), { 'stm_diff_loss': mem_diff_loss, 'policy_consistency_loss': interstep_kl_loss, 'stm_cosine_sim_loss': mem_sim }

    def kl_penalty(self, ref_log_probs: torch.Tensor, new_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kl = (ref_log_probs.exp() * (ref_log_probs - new_log_probs)) * mask
        kl_loss = (kl.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)).mean()
        return kl_loss

    def policy_consistency_loss(self, prev_step_log_probs: torch.Tensor, this_step_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. Apply mask to both distributions
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_prev_log_probs = prev_step_log_probs * mask_expanded
        masked_current_probs = this_step_probs * mask_expanded

        # 2. Flatten while preserving mask
        batch_size, seq_len, vocab_size = prev_step_log_probs.shape
        flat_prev_log_probs = masked_prev_log_probs.view(-1, vocab_size)
        flat_current_probs = masked_current_probs.view(-1, vocab_size)

        # 3. Filter out completely masked positions
        valid_indices = mask.flatten().bool()
        valid_prev_log_probs = flat_prev_log_probs[valid_indices]
        valid_current_probs = flat_current_probs[valid_indices]

        # 4. Compute KL divergence only for valid positions
        if len(valid_prev_log_probs) > 0:
            kl_loss = self.policy_consistency_loss_fn(
                valid_prev_log_probs,
                valid_current_probs,
            )
        else:
            kl_loss = torch.tensor(0.0).to(mask.device)

        return kl_loss

    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_gae:
            advantages, ref_values = self._compute_gae(rewards[:-1], values[:-1], values[-1], dones[:-1])
        else:
            advantages = rewards - values
            ref_values = rewards

        if self.use_distributed_advantage_norm:
            mean_advantage = distributed_mean(advantages.mean())
            std_advantage = distributed_mean(advantages.std())
            normalized_advantages = (advantages - mean_advantage) / (std_advantage + 1e-8)
        else:
            if self.use_gae:
                normalize_mask = ~dones.unsqueeze(-1)[:-1, :].expand_as(advantages)
                if self.advantage_mode == 'all':
                    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                elif self.advantage_mode == 'batch':
                    normalized_advantages = self.normalize_advantages_flatten(advantages, normalize_mask)
                else:
                    normalized_advantages = self.normalize_advantages_per_step_with_mask(advantages, normalize_mask)
            else:
                if self.advantage_mode == 'all':
                    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                elif self.advantage_mode == 'batch':
                    normalize_mask = ~dones.unsqueeze(-1).expand_as(advantages)
                    normalized_advantages = self.normalize_advantages_flatten(advantages, normalize_mask)
                else:
                    normalized_advantages = self.normalize_advantages_per_step(advantages)

        return normalized_advantages, ref_values

    def normalize_advantages_per_step_with_mask(self, advantages: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8,
                                      clip: float = 10.0) -> torch.Tensor:
        """
        advantages: [T, B]
        mask: [T, B]
        Normalizes along batch axis for each timestep t separately.
        """
        advantages = advantages.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        B, T = advantages.shape
        out = torch.zeros_like(advantages)
        for t in range(T):
            valid = mask[:, t].bool()
            if valid.sum() == 0:
                continue
            vals = advantages[valid, t]
            mean = vals.mean()
            std = vals.std(unbiased=False)
            normed = (vals - mean) / (std + eps)
            normed = torch.clamp(normed, -clip, clip)
            out[valid, t] = normed

        out = out.transpose(0, 1).contiguous()
        return out

    def normalize_advantages_per_step(self, advantages: torch.Tensor, eps: float = 1e-8,
                                      clip: float = 10.0) -> torch.Tensor:
        """
        advantages: [T, B]
        mask: [T, B]
        Normalizes along batch axis for each timestep t separately.
        """

        T, B = advantages.shape
        out = torch.zeros_like(advantages)
        for t in range(T):
            vals = advantages[t, :]
            mean = vals.mean()
            std = vals.std(unbiased=False)
            normed = (vals - mean) / (std + eps)
            normed = torch.clamp(normed, -clip, clip)
            out[t, :] = normed

        return out

    def normalize_advantages_flatten(self, advantages: torch.Tensor, mask: torch.Tensor, eps=1e-8, clip=10.0) -> torch.Tensor:
        """
        advantages: [T, B]  (we'll handle both)
        mask: same shape, 0/1 for valid positions
        Returns normalized advantages same shape.
        """
        advantages = advantages.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        # flatten valid entries
        flat = advantages[mask.bool()]
        if flat.numel() == 0:
            return advantages  # nothing to normalize

        mean = flat.mean()
        std = flat.std(unbiased=False)  # population std
        flat_norm = (flat - mean) / (std + eps)
        flat_norm = torch.clamp(flat_norm, -clip, clip)

        # put back
        out = advantages.clone()
        out[mask.bool()] = flat_norm

        out = out.transpose(0, 1).contiguous()
        return out

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     last_value: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_len, batch_size = rewards.shape
        advantages = torch.zeros_like(rewards, device=rewards.device)
        last_advantage = 0
        next_value = last_value
        dones = dones.float()

        for t in reversed(range(trajectory_len)):
            # Calculate delta from rewards, stored next_value, masked by stored next_done, and values
            delta = rewards[t] + self.gae_gamma * next_value * (1 - dones[t]) - values[t]
            # Calculate advantages based on delta, gamma/lambda factors and last advantage, masked by current done flags
            advantages[t] = delta + self.gae_gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            # Store current step data as last_advantage, next_done and next_value, for the next iteration step
            last_advantage = advantages[t]
            next_value = values[t]

        # Calculate reference returns, based on advantages and values, and return them with advantages for critic update
        returns = advantages + values
        return advantages, returns
