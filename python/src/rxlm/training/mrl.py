import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, TypedDict, Union, TypeAlias, Literal, Callable, Any
from enum import Enum
import random, os
from ..transformers.sampler import BatchSampler
from .callbacks import MrlTrainerCallback
from .dataset import MrlCurriculumDataset
from .utils import smart_concat, smart_concat_critic_states, TokenizedDict, get_gradient_norms
from .rl import RlAlgorithm
from .reward import MrlRewardMode, MrlRewardModel
from .models import MrlActorAction, MrlActorModel, MrlCriticModel
from .ddp import get_os_ddp_config, distributed_mean


class MrlConfig(TypedDict):
    lr: float
    separate_memory_lr: Optional[bool]
    memory_lr: Optional[float]
    critic_lr: float
    critic_encoder_lr: Optional[float]
    encoder_lr: Optional[float]
    memory_attn_lr: Optional[float]
    max_seq_len: int
    critic_max_len: int
    weight_decay: Optional[float]
    critic_weight_decay: Optional[float]
    update_epochs: int
    pad_token_id: int
    end_token_id: int
    answer_token_id: int
    callbacks: Optional[list[MrlTrainerCallback]]
    use_moe_aux_loss: Optional[bool]
    moe_aux_loss_scale: Optional[float]
    freeze_embeddings: Optional[bool]
    embedding_lr: Optional[float]
    use_memory_warmup: Optional[bool]
    hard_warmup: Optional[bool]
    debug_mode: Optional[bool]
    debug_interval: Optional[int]
    clamp_logits: Optional[float]
    max_grad_norm: Optional[float]
    use_self_attn_cache: Optional[bool]
    log_probs_source: Optional[Literal['collect', 'update']]


class MrlStrategy(Enum):
    SINGLE_STEP_STRATEGY = 1
    MULTI_STEP_STRATEGY = 2
    LONG_RANGE_STRATEGY = 3


UnfreezeStrategyFn = Callable[[int, Any], None]
UnfreezeItem = Union[int, tuple[int, float]]
UnfreezeEpochsStrategy: TypeAlias = Union[int, tuple[UnfreezeItem, UnfreezeItem, int], UnfreezeStrategyFn]

class SamplerConfig(TypedDict):
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]


class MrlTrajectoryStep(TypedDict):
    state: tuple[TokenizedDict, TokenizedDict, TokenizedDict]
    action: TokenizedDict
    log_probs: torch.Tensor
    ref_log_probs: Optional[torch.Tensor]
    reward: torch.Tensor
    reference: TokenizedDict
    done: bool


class MrlTrajectoryEpisode(TypedDict):
    reset_stm: bool
    steps: list[MrlTrajectoryStep]


class MrlReplayBuffer:
    def __init__(self, size: int, update_size: int, max_age: int = 3):
        self.size = size
        self.update_size = update_size
        self.max_age = max_age
        self.trajectories = []
        self.initialized = False

    def __getitem__(self, idx: int) -> MrlTrajectoryEpisode:
        return self.trajectories[idx]['episode']

    def __call__(self, collected: list[MrlTrajectoryEpisode]):
        if self.initialized:
            return self.update(collected)
        else:
            return self.initialize(collected)

    def reset(self):
        self.trajectories = []
        self.initialized = False

    def _process_episode(self, episode: MrlTrajectoryEpisode):
        steps_scores = [(step['reward'].mean() - step['reward'].std()).item() for step in episode['steps']]

        return {
            'episode': episode,
            'score': sum(steps_scores) / len(steps_scores),
        }

    def initialize(self, collected: list[MrlTrajectoryEpisode]):
        # 1. Init - get all collected episode scores (reward mean - reward std)
        processed = [{ **self._process_episode(ep), 'age': 0 } for ep in collected]
        # 2. Sort collected episodes by highest score
        processed.sort( key=lambda ep: ep['score'], reverse=True)
        # 3. Save N episodes with the highest score
        self.trajectories = processed[:self.size]
        # 4. Shuffle episodes for next run
        self.trajectories.sort(key=lambda _: random.random())
        # 5. Set Buffer as initialized
        self.initialized = True

    def update(self, collected: list[MrlTrajectoryEpisode]):
        # 1. Update - increment age of buffered episodes
        self.trajectories = [{ **traj, 'age': traj['age'] + 1 } for traj in self.trajectories]
        # 2. Sort buffered episodes by lowest score
        self.trajectories.sort( key=lambda ep: ep['score'])
        # 3. Remove the oldest episodes (starting from lower score)
        filtered, removed  = [], 0
        for traj in self.trajectories:
            if traj['age'] >= self.max_age and removed < self.update_size:
                removed += 1
            else:
                filtered.append(traj)

        # 4. When there wasn't enough old episodes, remove those with the lowest scores
        if removed < self.update_size:
            filtered = filtered[self.update_size - removed:]

        # 5. Get processed episodes from new collected ones
        processed_collected = [{ **self._process_episode(ep), 'age': 0 } for ep in collected[self.size:]]
        # 6. Sort collected episodes by the highest scores
        processed_collected.sort( key=lambda ep: ep['score'], reverse=True)
        # 7. Save filtered buffered and new collected episodes as new buffered episodes
        self.trajectories = (filtered + processed_collected[:self.update_size])
        # 8. Shuffle episodes for the next run
        self.trajectories.sort(key=lambda _: random.random())
        # 9. Assert correct size of the buffer
        assert len(self.trajectories) == self.size


class CurriculumConfig(TypedDict):
    steps: int
    epochs: int
    dataset: MrlCurriculumDataset
    eval_dataset: Optional[MrlCurriculumDataset]
    callbacks: Optional[list[MrlTrainerCallback]]
    strategy: MrlStrategy
    unfreeze_epoch: Optional[UnfreezeEpochsStrategy]
    random_resets: Optional[bool]
    random_resets_from: Optional[int]
    random_resets_ratio: Optional[float]
    reward_model: Optional[MrlRewardModel]
    separate_memory_lr: Optional[bool]
    lr: Optional[float]
    memory_lr: Optional[float]
    critic_lr: Optional[float]
    critic_encoder_lr: Optional[float]
    encoder_lr: Optional[float]
    memory_attn_lr: Optional[float]
    weight_decay: Optional[float]
    critic_weight_decay: Optional[float]
    update_epochs: Optional[int]
    freeze_embeddings: Optional[bool]
    embedding_lr: Optional[float]
    teacher_forcing: Optional[bool]
    use_memory_warmup: Optional[bool]
    hard_warmup: Optional[bool]
    replay_buffer: Optional[MrlReplayBuffer]


OptimField: TypeAlias = Literal[
    'lr', 'critic_lr', 'weight_decay', 'critic_weight_decay', 'separate_memory_lr',
    'memory_lr', 'encoder_lr', 'memory_attn_lr'
]

class MRLTrainer:
    def __init__(
            self,
            actor: MrlActorModel,
            critic: MrlCriticModel,
            reference_model: MrlActorModel,
            reward: MrlRewardModel,
            device: torch.device,
            config: MrlConfig,
            rl_algorithm: RlAlgorithm,
            sampler_config: Optional[SamplerConfig] = None,
            log_dir: str = None,
            use_ddp: bool = False,
            use_amp: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        """
        Trainer for Memory Reinforcement Learning (MRL) algorithm for reactive models and Attention-Based Memory System.

        Args:
            actor (MrlActorModel): MRL Actor model with encoder, decoder and memory attention.
            critic (MrlCriticModel): MRL Critic network for advantage estimation.
            reward (MrlRewardModel): MRL Reward model or extension.
            device (torch.device): Device used for training.
            config (MrlConfig): Configuration dictionary with hyperparameters.
            rl_algorithm (RlAlgorithm): Reinforcement Learning algorithm (currently only PPO available).
            sampler_config (SamplerConfig): Sampler configuration.
            log_dir (str): Log directory for TensorBoard logs.
            use_ddp (bool): Use Distributed Data Parallel mode.
            use_amp (bool): Use AMP Autocast for training.
            dtype (torch.dtype): Data type used in training - in AMP mode it's auto cast, otherwise data and model are transformed to this type
        """
        self.actor = actor
        self.critic = critic
        self.reference_model = reference_model

        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

        self.shared_reward_model = reward
        self.reward = reward
        self.device = device
        self.max_seq_len = config.get('max_seq_len', 256)
        self.critic_max_len = config.get('critic_max_len', 512)
        self.use_moe_aux_loss = config.get('use_moe_aux_loss', False)
        self.moe_aux_loss_scale = config.get('moe_aux_loss_scale', 0.01)
        self.shared_freeze_embeddings = config.get('freeze_embeddings', False)
        self.freeze_embeddings = self.shared_freeze_embeddings
        self.debug_mode = config.get('debug_mode', False)
        self.debug_interval = config.get('debug_interval', 10)
        self.clamp_logits = config.get('clamp_logits', None)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.log_probs_source = config.get('log_probs_source', 'collect')

        self.base_memory_warmup_config = {
            'use_memory_warmup': config.get('use_memory_warmup', False),
            'hard_warmup': config.get('hard_warmup', False)
        }
        self.use_memory_warmup = self.base_memory_warmup_config['use_memory_warmup']
        self.hard_warmup = self.base_memory_warmup_config['hard_warmup']

        self.use_self_attn_cache = config.get('use_self_attn_cache', True)
        # Internal update epochs config
        self.shared_update_epochs = config.get('update_epochs', 10)
        self.update_epochs = self.shared_update_epochs

        # Move models to device
        if use_amp:
            self.actor.to(self.device)
            self.critic.to(self.device)
        else:
            self.actor.to(self.device, dtype=dtype)
            self.critic.to(self.device, dtype=dtype)

        # Batch Sampler for answer generation
        self.generator = None
        self.sampler_config = SamplerConfig(
            temperature=1.0,
            top_k=None,
            top_p=None,
        ) if sampler_config is None else sampler_config

        self.pad_token_id = config.get('pad_token_id', 0)
        self.end_token_id = config.get('end_token_id', 3)
        self.answer_token_id = config.get('answer_token_id', 6)

        self.use_ddp = use_ddp
        self.use_amp = use_amp
        self.dtype = dtype

        self.separate_memory_lr = config.get('separate_memory_lr', False)

        if self.separate_memory_lr:
            self.base_optim_config = {
                'lr': config.get('lr', 3e-4),
                'memory_lr': config.get('memory_lr', 5e-4),
                'critic_lr': config.get('critic_lr', 1e-4),
                'weight_decay': config.get('weight_decay', 0.01),
                'critic_weight_decay': config.get('critic_weight_decay', 0.01),
                'critic_encoder_lr': config.get('critic_encoder_lr', config.get('critic_lr', 1e-4)),
                'embedding_lr': config.get('embedding_lr', config.get('lr', 3e-4)),
                'encoder_lr': config.get('encoder_lr', config.get('lr', 3e-4)),
                'memory_attn_lr': config.get('memory_attn_lr', config.get('memory_lr', 5e-4)),
            }
        else:
            self.base_optim_config = {
                'lr': config.get('lr', 3e-4),
                'critic_lr': config.get('critic_lr', 1e-4),
                'weight_decay': config.get('weight_decay', 0.01),
                'critic_weight_decay': config.get('critic_weight_decay', 0.01),
                'critic_encoder_lr': config.get('critic_encoder_lr', config.get('critic_lr', 1e-4)),
                'embedding_lr': config.get('embedding_lr', config.get('lr', 3e-4)),
                'encoder_lr': config.get('encoder_lr', config.get('lr', 3e-4)),
            }

        self.optim_config = self.base_optim_config

        self.optimizer, self.critic_optimizer = self._init_optimizers(**self.optim_config)

        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        self.critic_scaler = torch.amp.GradScaler() if self.use_amp else None

        # TensorBoard Writer
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir) if log_dir else None

        self.global_step = self._init_steps()
        self.epoch_step = self._init_steps()
        self.stage_step = self._init_steps()

        self.rl_algorithm = rl_algorithm

        # Dynamic fields, updated for each curriculum step
        self.curriculum_steps = 0
        self.train_dataset = None
        self.eval_dataset = None
        self.random_resets_ratio = 0.0
        self.strategy = None
        self.shared_callbacks = config.get('callbacks', [])
        self.callbacks = []
        self.global_epoch = 0
        self.global_epochs_count = 0
        self.teacher_forcing = False
        self.replay_buffer = None

    def _init_optimizers(
            self,
            lr: float,
            critic_lr: float,
            weight_decay: float,
            critic_weight_decay: float,
            critic_encoder_lr: float,
            embedding_lr: float,
            encoder_lr: float,
            memory_lr: Optional[float] = None,
            memory_attn_lr: Optional[float] = None,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        if memory_lr is not None:
            optimizer = torch.optim.AdamW([
                {'params': self.actor.embedding_parameters(), 'lr': embedding_lr},
                {'params': self.actor.encoder.not_memory_parameters(), 'lr': encoder_lr},
                {'params': self.actor.memory_attention_parameters(), 'lr': memory_attn_lr},
                {'params': self.actor.decoder.memory_parameters(), 'lr': memory_lr},
                {'params': self.actor.decoder.not_memory_parameters(), 'lr': lr},
            ],
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW([
                {'params': self.actor.embedding_parameters(), 'lr': embedding_lr},
                {'params': self.actor.embedding_parameters(), 'lr': embedding_lr},
                {'params': self.actor.encoder.not_memory_parameters(), 'lr': encoder_lr},
                {'params': self.actor.memory_attention_parameters(), 'lr': lr},
                {'params': self.actor.decoder.memory_parameters(), 'lr': lr},
                {'params': self.actor.decoder.not_memory_parameters(), 'lr': lr},
            ],
                weight_decay=weight_decay,
            )

        critic_optimizer = torch.optim.AdamW(
            [
                {'params': self.critic.head_parameters(), 'lr': critic_lr},
                {'params': self.critic.encoder_parameters(), 'lr': critic_encoder_lr},
            ],
            weight_decay=critic_weight_decay,
        )

        return optimizer, critic_optimizer

    def _init_steps(self):
        return {
            'collect': 0,
            'train': 0,
            'eval': 0,
        }

    def _increment_steps(self, step_type: str):
        self.global_step[step_type] += 1
        self.epoch_step[step_type] += 1
        self.stage_step[step_type] += 1

    def reset_stm(self, force: bool = False) -> bool:
        """Reset Short-Term Memory state with random reset ratio."""
        if force:
            self.actor.reset_memory()
            return True
        elif self.random_resets_ratio == 1.0:
            self.actor.reset_memory()
            return True
        else:
            rng = random.random()
            if rng <= self.random_resets_ratio:
                self.actor.reset_memory()
                return True
            else:
                return False

    def encode_and_update_stm(self, query: TokenizedDict, answer: TokenizedDict, for_ref_model: bool = False):
        """Encode interaction and update STM."""
        # 1. Encode data and update memory - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                # 2. Concatenate batch of queries and answers (they are already on training device)
                inputs = smart_concat(query, answer, self.max_seq_len, self.pad_token_id)
                # 3. Encode data and update STM
                if for_ref_model:
                    self.reference_model(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                               action=MrlActorAction.UPDATE)
                else:
                    self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.UPDATE)
        else:
            # 2. Concatenate batch of queries and answers (they are already on training device)
            inputs = smart_concat(query, answer, self.max_seq_len, self.pad_token_id)
            # 3. Encode data and update STM
            if for_ref_model:
                self.reference_model(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                     action=MrlActorAction.UPDATE)
            else:
                self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'], action=MrlActorAction.UPDATE)

    def generate_answer(self, query: TokenizedDict) -> tuple[TokenizedDict, torch.Tensor]:
        """Generate response using batch sampler with decoder."""
        # 1. Generate answer with BatchSampler - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                input_ids, attention_mask, log_probs = self.generator(
                    query['input_ids'],
                    query['attention_mask'],
                    max_gen_len=self.max_seq_len,
                    dtype=self.dtype,
                    **self.sampler_config,
                )
        else:
            input_ids, attention_mask, log_probs = self.generator(
                query['input_ids'],
                query['attention_mask'],
                max_gen_len=self.max_seq_len,
                dtype=self.dtype,
                **self.sampler_config,
            )
        # 2. Convert generated answer to TokenizedDict
        generated_answer: TokenizedDict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        return generated_answer, log_probs

    def _calculate_reward(self, generated: TokenizedDict, reference: TokenizedDict,
                          saved_query: TokenizedDict, saved_answer: TokenizedDict,
                          mode: MrlRewardMode = MrlRewardMode.STANDARD,
                          prev_data: tuple[TokenizedDict, TokenizedDict] = None):
        saved_interaction = smart_concat(saved_query, saved_answer, max_length=self.max_seq_len,
                                         pad_token_id=self.pad_token_id)
        prev_data = smart_concat(prev_data[0], prev_data[1], self.max_seq_len,
                                 self.pad_token_id) if prev_data is not None else None
        return self.reward(generated, reference, saved_interaction, mode=mode, prev_data=prev_data), saved_interaction

    def compute_reward(self, generated: TokenizedDict, reference: TokenizedDict,
                       saved_data: tuple[TokenizedDict, TokenizedDict], query: TokenizedDict, mode: MrlRewardMode = MrlRewardMode.STANDARD,
                       eval_mode: bool = False, prev_data: tuple[TokenizedDict, TokenizedDict] = None) -> torch.Tensor:
        """Compute reward based on memory retention (e.g., BLEU-4)."""
        # 1. Move sequences to GPU for reward calculation
        saved_query, saved_answer = self._move_multiple_batches(*saved_data)
        reference = self._move_batch(reference)
        prev_data = self._move_multiple_batches(*prev_data) if prev_data is not None else None

        # 2. Concat saved (previous) interaction and calculate reward using generated sequence, reference and saved data - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                reward, saved_interaction = self._calculate_reward(
                    generated, reference, saved_query, saved_answer,
                    mode=mode, prev_data=prev_data
                )
        else:
            reward, saved_interaction = self._calculate_reward(
                generated, reference, saved_query, saved_answer,
                mode=mode, prev_data=prev_data
            )

        # 3. Run 'on reward' callbacks
        for cb in self.callbacks:
            cb.on_reward(self.actor, reward.tolist(), generated, reference, saved_interaction, query, eval_mode)
        # 4. Return rewards for batch
        return reward

    def _move_batch(self, batch: TokenizedDict) -> TokenizedDict:
        if self.use_amp:
            return {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
            }
        else:
            return {
                'input_ids': batch['input_ids'].to(self.device, dtype=self.dtype),
                'attention_mask': batch['attention_mask'].to(self.device, dtype=self.dtype),
            }

    def _move_multiple_batches(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._move_batch(batch) for batch in batches]

    def _cpu_detach(self, batch: TokenizedDict) -> TokenizedDict:
        return {
            'input_ids': batch['input_ids'].detach().cpu(),
            'attention_mask': batch['attention_mask'].detach().cpu(),
        }

    def _batch_detach(self, batch: TokenizedDict) -> TokenizedDict:
        return {
            'input_ids': batch['input_ids'].detach(),
            'attention_mask': batch['attention_mask'].detach(),
        }

    def _cpu_detach_multiple(self, *batches: TokenizedDict) -> list[TokenizedDict]:
        return [self._cpu_detach(batch) for batch in batches]

    def _collect_writer(self, avg_reward: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar('Collect/episode reward (global)', avg_reward, self.global_step['collect'])
            self.writer.add_scalar(f'Collect/episode reward (steps: {self.curriculum_steps}, epoch: {epoch})',
                                   avg_reward, self.epoch_step['collect'])
            self.writer.add_scalar(f'Collect/episode reward (steps: {self.curriculum_steps})', avg_reward,
                                   self.stage_step['collect'])

    def _hard_memory_warmup(self, query: TokenizedDict, answer: TokenizedDict):
        # 1. Encode data and update memory - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                # 2. Concatenate batch of queries and answers (they are already on training device)
                inputs = smart_concat(query, answer, self.max_seq_len, self.pad_token_id)
                # 3. Encode data and update STM
                _, ed = self.actor.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                stm_initial_state = self.actor.memory_attention.model.stm.memory
                self.actor.memory_attention.model.stm.update_all((ed + stm_initial_state) / 2)
        else:
            # 2. Concatenate batch of queries and answers (they are already on training device)
            inputs = smart_concat(query, answer, self.max_seq_len, self.pad_token_id)
            # 3. Encode data and update STM
            _, ed = self.actor.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            stm_initial_state = self.actor.memory_attention.model.stm.memory
            self.actor.memory_attention.model.stm.update_all((ed + stm_initial_state) / 2)

    def memory_warmup(self, query: TokenizedDict, answer: TokenizedDict):
        if self.use_memory_warmup:
            with torch.no_grad():
                if self.hard_warmup:
                    self._hard_memory_warmup(query, answer)
                else:
                    self.encode_and_update_stm(query, answer)

    def collect_trajectories(self, dataloader: DataLoader, epoch: int, batch_size: int) -> list[MrlTrajectoryEpisode]:
        """Collect trajectories for PPO for current curriculum step."""
        # 1. Init trajectories list
        trajectories = []

        with torch.no_grad():
            # 2. Collect episode trajectories for all batches in dataset
            for batch_idx, batch in enumerate(dataloader):
                if self.replay_buffer is not None and self.replay_buffer.initialized and batch_idx < self.replay_buffer.size and self.replay_buffer[batch_idx] is not None:
                    self._increment_steps('collect')
                    episode_trajectory = self.replay_buffer[batch_idx]
                    trajectories.append(episode_trajectory)

                    episode_rewards = [step['reward'] for step in episode_trajectory['steps']]

                    mean_episode_reward = torch.stack(episode_rewards).mean().item()

                    self._collect_writer(mean_episode_reward, epoch)

                    print(f'Batch {batch_idx} from replay buffer')
                    # 15. Run "on episode collected" callbacks
                    for cb in self.callbacks:
                        cb.on_episode_collected(self.actor, batch_idx, episode_trajectory, mean_episode_reward)
                else:
                    if batch['query']['input_ids'].size(0) == batch_size:
                        self._increment_steps('collect')
                        # 3. Reset Short-Term Memory state (with random reset ratio - sometimes it will be good to build memory
                        # state from existing one, instead of new random one)
                        reset_done = self.reset_stm()

                        # 4. Reset reward prev data running mean - it's calculated for multistep retention, we have to reset it before episode
                        self.reward.reset_running_mean()

                        # 5. Get first batch of interactions (data to save) and follow-up interactions for current episode, based on curriculum step
                        first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']
                        interactions = interactions[:self.curriculum_steps]
                        interactions_len = len(interactions)

                        first_interaction = self._move_multiple_batches(first_query, first_answer)

                        if reset_done:
                            self.memory_warmup(*first_interaction)
                        # 6. Encode and update STM with data to save from first interaction
                        self.encode_and_update_stm(*first_interaction)

                        # 7. Save first interaction as data to save (for trajectory state)
                        query, answer = first_query, first_answer

                        # 8. Run training strategy for follow-up interactions
                        episode_steps = []
                        episode_rewards = []

                        prev_interaction = None

                        for i, interaction in enumerate(interactions):
                            # 9. Generate batch of answers based on batch of follow-up queries
                            next_query = self._move_batch(interaction['query'])
                            generated_answer, log_probs = self.generate_answer(next_query)

                            is_last_interaction = (i + 1) == interactions_len

                            detached_answer = self._batch_detach(generated_answer)

                            # 10. Depending on strategy compute reward
                            if self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0:
                                # a) long-range - first interaction - change topic - negative reward (it shouldn't include saved data)
                                reward = self.compute_reward(
                                    detached_answer, interaction['answer'], (query, answer),
                                    interaction['query'], mode=MrlRewardMode.NEGATIVE
                                )
                            elif self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and is_last_interaction:
                                # b) long-range - last interaction - first interaction topic - long-range reward (it should include content from first interaction)
                                reward = self.compute_reward(
                                    detached_answer, interaction['answer'], (first_query, first_answer),
                                    interaction['query'], mode=MrlRewardMode.LONG_RANGE, prev_data=prev_interaction
                                )
                            else:
                                # c) standard reward - generated answer should include some content from previous interaction (saved data), like reference answer
                                reward = self.compute_reward(
                                    detached_answer, interaction['answer'], (query, answer),
                                    interaction['query'], mode=MrlRewardMode.STANDARD, prev_data=prev_interaction
                                )

                            # 11. Update STM with generated response (except last interaction, it's not needed)
                            if not is_last_interaction:
                                self.encode_and_update_stm(
                                    next_query,
                                    self._move_batch(interaction['answer']) if self.teacher_forcing else generated_answer
                                )  # update with generated_answer on GPU

                            cpu_detached_answer = self._cpu_detach(generated_answer)  # detach and keep states on CPU

                            # 12. Store trajectory step
                            trajectory: MrlTrajectoryStep = {
                                'state': (query, answer, interaction['query']),
                                'action': cpu_detached_answer,
                                'log_probs': log_probs.detach().cpu(),
                                'ref_log_probs': None,
                                'reward': reward.detach().cpu(),
                                'reference': interaction['answer'],
                                'done': is_last_interaction,
                            }
                            episode_steps.append(trajectory)
                            episode_rewards.append(reward)

                            # 13. Set previous and current interaction query and generated answer (batches), as saved data for next interaction
                            if not (self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0):
                                prev_interaction = (query, answer)
                            query, answer = interaction['query'], (interaction['answer'] if self.teacher_forcing else cpu_detached_answer)

                        # 14. Append full batched episode (number of steps depends on curriculum stage) to trajectories
                        episode_trajectory: MrlTrajectoryEpisode = {
                            'reset_stm': reset_done,
                            'steps': episode_steps,
                        }
                        trajectories.append(episode_trajectory)

                        mean_episode_reward = torch.stack(episode_rewards).mean().item()

                        self._collect_writer(mean_episode_reward, epoch)

                        # 15. Run "on episode collected" callbacks
                        for cb in self.callbacks:
                            cb.on_episode_collected(self.actor, batch_idx, episode_trajectory, mean_episode_reward)

        return trajectories

    def _critic_writer(self, critic_loss: float, epoch: int, warmup_step: Optional[int] = None):
        if self.writer is not None:
            if epoch == -1 and warmup_step is not None:
                self.writer.add_scalar('Loss/critic warmup', critic_loss, warmup_step)
            else:
                self.writer.add_scalar('Loss/critic (global)', critic_loss, self.global_step['train'])
                self.writer.add_scalar(f'Loss/critic (steps: {self.curriculum_steps}, epoch: {epoch})', critic_loss,
                                       self.epoch_step['train'])
                self.writer.add_scalar(f'Loss/critic (steps: {self.curriculum_steps})', critic_loss,
                                       self.stage_step['train'])

    def _rl_writer(self, policy_loss: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar('Loss/policy (global)', policy_loss, self.global_step['train'])
            self.writer.add_scalar(f'Loss/policy (steps: {self.curriculum_steps}, epoch: {epoch})', policy_loss,
                                   self.epoch_step['train'])
            self.writer.add_scalar(f'Loss/policy (steps: {self.curriculum_steps})', policy_loss,
                                   self.stage_step['train'])

    def update_critic(self, state: tuple[TokenizedDict, TokenizedDict, TokenizedDict], ref_values: torch.Tensor,
                      epoch: int, warmup_step: Optional[int] = None) -> float:
        # 1. Reset critic gradients
        self.critic_optimizer.zero_grad()

        # 2. Update critic - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                # 2.1 Concat states and calculate critic loss
                critic_state = smart_concat_critic_states(*state, max_length=self.critic_max_len,
                                                          pad_token_id=self.pad_token_id)
                values = self.critic(critic_state['input_ids'], attention_mask=critic_state['attention_mask']).squeeze()
                critic_loss = self.rl_algorithm.critic_loss(values, ref_values.detach())
            # 2.2 Run backpropagation with scaler
            self.critic_scaler.scale(critic_loss).backward()
            # 2.3 Unscale and clip gradients
            self.critic_scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, error_if_nonfinite=False)
            # 2.4 Run scaled optimization step
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()
        else:
            # 2.1 Concat states and calculate critic loss
            critic_state = smart_concat_critic_states(*state, max_length=self.critic_max_len,
                                                      pad_token_id=self.pad_token_id)
            values = self.critic(critic_state['input_ids'], attention_mask=critic_state['attention_mask']).squeeze()
            critic_loss = self.rl_algorithm.critic_loss(values, ref_values.detach())
            # 2.2 Run backpropagation
            critic_loss.backward()
            # 2.3 Clip gradients
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, error_if_nonfinite=False)
            # 2.4 Run optimizer step
            self.critic_optimizer.step()
        # 3. Get float loss value for callbacks/writer
        critic_loss_item = critic_loss.item()

        # 4. Write to TensorBoard
        self._critic_writer(critic_loss_item, epoch, warmup_step=warmup_step)

        # 5. Run "on critic updated" callbacks
        for cb in self.callbacks:
            cb.on_critic_updated(self.actor, self.critic, epoch, self.epoch_step['train'] if warmup_step is None else warmup_step, critic_loss_item)
        # 6. Return loss item
        return critic_loss_item

    def _moe_aux_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_aux_loss:
            return main_loss

        actor = next(self.actor.children()) if isinstance(self.actor, DistributedDataParallel) else self.actor

        router_loss = actor.moe_router_loss()
        if router_loss is not None:
            return main_loss + self.moe_aux_loss_scale * router_loss
        else:
            return main_loss

    def _clip_actor_grad_norms(self):
        # Encoder with embedding
        torch.nn.utils.clip_grad_norm_(
            self.actor.encoder.parameters(),
            max_norm=self.max_grad_norm, error_if_nonfinite=False
        )
        # Decoder
        torch.nn.utils.clip_grad_norm_(
            self.actor.decoder.memory_parameters() + self.actor.decoder.not_memory_parameters(),
            max_norm=self.max_grad_norm, error_if_nonfinite=False
        )
        # Memory attention
        torch.nn.utils.clip_grad_norm_(
            self.actor.memory_attention.parameters(),
            max_norm=self.max_grad_norm, error_if_nonfinite=False
        )

    def _log_gradients(self, logits: torch.Tensor):
        print(
            f"----- Returned logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        encoder_total, encoder_mean = get_gradient_norms(self.actor.encoder.parameters())
        decoder_total, decoder_mean = get_gradient_norms(self.actor.decoder.parameters())
        mem_att_total, mem_att_mean = get_gradient_norms(self.actor.memory_attention.parameters())
        print(f"----- Encoder grad norm - total: {encoder_total:.6f}, mean: {encoder_mean:.6f}")
        print(f"----- Decoder grad norm - total: {decoder_total:.6f}, mean: {decoder_mean:.6f}")
        print(f"----- Memory attention grad norm - total: {mem_att_total:.6f}, mean: {mem_att_mean:.6f}")

        dec_mem_total, dec_mem_mean = get_gradient_norms(self.actor.decoder.memory_parameters())
        dec_not_mem_total, dec_not_mem_mean = get_gradient_norms(self.actor.decoder.not_memory_parameters())

        print(f"----- Decoder memory params grad norm - total: {dec_mem_total:.6f}, mean: {dec_mem_mean:.6f}")
        print(f"----- Decoder not memory params grad norm - total: {dec_not_mem_total:.6f}, mean: {dec_not_mem_mean:.6f}")

        if self.writer is not None:
            self.writer.add_scalar('Gradient/encoder', encoder_mean, self.global_step['train'])
            self.writer.add_scalar('Gradient/decoder', decoder_mean, self.global_step['train'])
            self.writer.add_scalar('Gradient/mem-att', mem_att_mean, self.global_step['train'])
            self.writer.add_scalar('Gradient/decoder memory', dec_mem_mean, self.global_step['train'])
            self.writer.add_scalar('Gradient/decoder not memory', dec_not_mem_mean, self.global_step['train'])

    def _log_stm_diff(self, stm_diff: torch.Tensor, step_idx: int):
        stm_update_diff = torch.sqrt(stm_diff).item()
        print(f'--- STM update diff in step {step_idx + 1}: {stm_update_diff:.6f}')
        self.writer.add_scalar('STM/update diff (all)', stm_update_diff, self.global_step['train'])
        self.writer.add_scalar(f'STM/memory diff (step {step_idx + 1})', stm_update_diff,
                               self.global_step['train'])

    def _log_stm_cosine_sim(self, stm_sim: torch.Tensor, step_idx: int):
        stm_sim = stm_sim.item()
        print(f'--- STM cosine sim in step {step_idx + 1}: {stm_sim:.6f}')
        self.writer.add_scalar('STM/cosine sim (all)', stm_sim, self.global_step['train'])
        self.writer.add_scalar(f'STM/cosine sim (step {step_idx + 1})', stm_sim,
                               self.global_step['train'])
    def update_actor(
            self,
            state: tuple[TokenizedDict, TokenizedDict, TokenizedDict],
            action: TokenizedDict,
            advantages: torch.Tensor,
            old_log_probs: torch.Tensor,
            ref_log_probs: torch.Tensor,
            epoch: int,
            step_idx: int,
            prev_step_log_probs: Optional[torch.Tensor] = None,
    ) -> tuple[float, torch.Tensor]:
        # 1. Reset actor gradients
        self.optimizer.zero_grad()
        # 2. Unpack state dicts
        query, answer, next_query = state

        # 3. Save initial detached STM state
        initial_stm = self.actor.memory_attention.model.stm.memory.clone().detach()
        # 4. Encode interaction and update STM on each step, to include encoder and memory attention gradients in loss
        self.encode_and_update_stm(query, answer)
        # 5. Get updated STM state
        updated_stm = self.actor.memory_attention.model.stm.memory.clone()

        # Remove last item from old_log_probs in 'collect' mode
        old_log_probs = old_log_probs[:, :-1] if self.log_probs_source == 'collect' else old_log_probs

        # 6. Update actor - with autocast on/off
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                # 6.1 Concatenate next query and action and get action logits from decoder
                inputs = smart_concat(next_query, action, max_length=self.max_seq_len,
                                      pad_token_id=self.pad_token_id)
                logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                    action=MrlActorAction.DECODE)
                if self.clamp_logits is not None:
                    logits = logits.clamp(min=-self.clamp_logits, max=self.clamp_logits)
                # 6.2 Calculate policy loss with selected algorithm
                policy_loss, this_step_log_probs, extras = self.rl_algorithm.policy_loss(
                    next_query, action, logits, old_log_probs, advantages,
                    initial_stm, updated_stm, ref_log_probs, step_idx, prev_step_log_probs=prev_step_log_probs
                )
                policy_loss = self._moe_aux_loss(policy_loss)

            # 6.3 Run backpropagation with scaler
            self.scaler.scale(policy_loss).backward(retain_graph=True)
            # 6.4 Unscale and clip gradient norms
            self.scaler.unscale_(self.optimizer)
            self._clip_actor_grad_norms()
            if self.debug_mode and self.epoch_step['train'] % self.debug_interval == 0:
                self._log_gradients(logits)
                if 'stm_diff_loss' in extras:
                    self._log_stm_diff(extras['stm_diff_loss'], step_idx)
                if 'stm_cosine_sim_loss' in extras:
                    self._log_stm_cosine_sim(extras['stm_cosine_sim_loss'], step_idx)
            # 6.5 Run scaled optimization step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 6.1 Concatenate next query and action and get action logits from decoder
            inputs = smart_concat(next_query, action, max_length=self.max_seq_len,
                                  pad_token_id=self.pad_token_id)
            logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                action=MrlActorAction.DECODE)
            if self.clamp_logits is not None:
                logits = logits.clamp(min=-self.clamp_logits, max=self.clamp_logits)
            # 6.2 Calculate policy loss with selected algorithm
            policy_loss, this_step_log_probs, extras = self.rl_algorithm.policy_loss(
                next_query, action, logits, old_log_probs, advantages,
                initial_stm, updated_stm, ref_log_probs, step_idx, prev_step_log_probs=prev_step_log_probs
            )
            policy_loss = self._moe_aux_loss(policy_loss)
            # 6.3 Run backpropagation
            policy_loss.backward(retain_graph=True)
            # 6.4 Clip gradient norms
            self._clip_actor_grad_norms()
            if self.debug_mode and self.epoch_step['train'] % self.debug_interval == 0:
                self._log_gradients(logits)
                if 'stm_diff_loss' in extras:
                    self._log_stm_diff(extras['stm_diff_loss'], step_idx)
                if 'stm_cosine_sim_loss' in extras:
                    self._log_stm_cosine_sim(extras['stm_cosine_sim_loss'], step_idx)
            # 6.5 Run scaled optimization step
            self.optimizer.step()
        # 7. Get float loss value for callbacks/writer
        policy_loss_item = policy_loss.item()

        # 8. Write to TensorBoard
        self._rl_writer(policy_loss_item, epoch)

        # 9. Run "on batch updated" callback
        for cb in self.callbacks:
            cb.on_batch_updated(self.actor, epoch, self.epoch_step['train'], policy_loss_item)

        # 10. Return loss item
        return policy_loss_item, this_step_log_probs

    def rl_step(self, trajectories: list[MrlTrajectoryEpisode], advantages: torch.Tensor, ref_values: torch.Tensor,
                epoch: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform PPO update step using trajectories."""
        # 1. Run update separately for episodes in trajectory - we have to reset memory before each episode, and update
        # memory, based on collected episode data
        all_losses = []
        critic_losses = []
        for episode_idx, episode in enumerate(trajectories):
            episode_steps = episode['steps']
            should_reset_stm = episode['reset_stm']

            # 2. Get advantages and reference values for current full episode (batch_size * episode_steps)
            num_steps = len(episode_steps)
            start = episode_idx * num_steps
            end = start + num_steps
            episode_critic_values = ref_values[start:end]
            episode_advantages = advantages[start:end]

            # 3. Reset memory for current batch episode
            if should_reset_stm:
                self.reset_stm(force=True)

            stored_prev_step_log_probs = None

            # 4. Run episode steps - each episode has number of steps depending on curriculum stage. Each step is run for all batch
            for step_idx, step in enumerate(episode_steps):
                self._increment_steps('train')
                # 5. Get and move to device collected states, action and log probs
                state, action, _, log_probs, ref_log_probs = step['state'], step['action'], step['reward'], step['log_probs'], step['ref_log_probs']
                query, answer, next_query = self._move_multiple_batches(*state)
                action = self._move_batch(action)
                log_probs = log_probs.to(self.device)
                ref_log_probs = ref_log_probs.to(self.device)

                # 6. Select advantages and reference values for current step (batch_size)
                step_critic_values = episode_critic_values[step_idx]
                step_advantages = episode_advantages[step_idx]

                self.actor.clone_reset_memory()

                if should_reset_stm and step_idx == 0:
                    self.memory_warmup(query, answer)

                # 7. Update critic
                critic_loss_item = self.update_critic((query, answer, next_query), step_critic_values, epoch)

                # 8. Accumulate critic loss for epoch callbacks
                critic_losses.append(critic_loss_item)

                prev_step_log_probs = stored_prev_step_log_probs if step_idx != 0 else None

                # 9. Update actor
                policy_loss_item, stored_prev_step_log_probs = self.update_actor(
                    (query, answer, next_query),
                    action, step_advantages, log_probs, ref_log_probs,
                    epoch, step_idx, prev_step_log_probs=prev_step_log_probs
                )

                all_losses.append(policy_loss_item)
        # 10. Return mean losses for epoch callbacks
        return torch.mean(torch.tensor(all_losses)), torch.mean(torch.tensor(critic_losses))

    def _critic_values_rewards_and_dones(self, trajectories: list[MrlTrajectoryEpisode]):
        flat_trajectories = [t for episode in trajectories for t in episode['steps']]
        values = torch.stack([
            self._critic_values(*self._move_multiple_batches(*t['state'])) for t in flat_trajectories
        ]).to(self.device)
        rewards = torch.stack([t['reward'] for t in flat_trajectories]).to(self.device)
        dones = torch.stack([torch.tensor(t['done']) for t in flat_trajectories]).to(self.device)
        return values, rewards, dones

    def _critic_values(self, *moved_state: tuple[TokenizedDict, TokenizedDict, TokenizedDict]) -> torch.Tensor:
        # 1. Calculate critic values
        with torch.no_grad():
            # 2. Get concatenated critic states
            inputs = smart_concat_critic_states(
                *moved_state,
                max_length=self.critic_max_len,
                pad_token_id=self.pad_token_id,
            )
            # 3. Calculate values for current batch
            return self.critic(inputs['input_ids'],
                               attention_mask=inputs['attention_mask']).squeeze()

    def get_pre_update_log_probs(self, trajectories: list[MrlTrajectoryEpisode], for_ref_model: bool = False) -> list[MrlTrajectoryEpisode]:
        new_trajectories = []

        if for_ref_model:
            print('Computing ref model log probs')
        else:
            print('Recomputing old log probs')

        with torch.no_grad():
            for episode_idx, episode in enumerate(trajectories):
                episode_steps = episode['steps']
                should_reset_stm = episode['reset_stm']

                if should_reset_stm:
                    self.reset_stm(force=True)

                new_episode_steps = []

                # 4. Run episode steps - each episode has number of steps depending on curriculum stage. Each step is run for all batch
                for step_idx, step in enumerate(episode_steps):
                    # self._increment_steps('train')
                    # 5. Get and move to device collected states, action and log probs
                    state, action = step['state'], step['action']
                    query, answer, next_query = self._move_multiple_batches(*state)
                    action = self._move_batch(action)

                    if for_ref_model:
                        self.reference_model.clone_reset_memory()
                    else:
                        self.actor.clone_reset_memory()

                    if should_reset_stm and step_idx == 0:
                        self.memory_warmup(query, answer)

                    self.encode_and_update_stm(query, answer, for_ref_model=for_ref_model)

                    if self.use_amp:
                        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                            # 6.1 Concatenate next query and action and get action logits from decoder
                            inputs = smart_concat(next_query, action, max_length=self.max_seq_len,
                                                  pad_token_id=self.pad_token_id)
                            if for_ref_model:
                                logits = self.reference_model(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                                action=MrlActorAction.DECODE)
                            else:
                                logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                                action=MrlActorAction.DECODE)
                            if self.clamp_logits is not None:
                                logits = logits.clamp(min=-self.clamp_logits, max=self.clamp_logits)
                            log_probs = self._get_answer_log_probs(next_query, action, logits)
                    else:
                        # 6.1 Concatenate next query and action and get action logits from decoder
                        inputs = smart_concat(next_query, action, max_length=self.max_seq_len,
                                              pad_token_id=self.pad_token_id)
                        if for_ref_model:
                            logits = self.reference_model(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                                          action=MrlActorAction.DECODE)
                        else:
                            logits = self.actor(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                                action=MrlActorAction.DECODE)
                        if self.clamp_logits is not None:
                            logits = logits.clamp(min=-self.clamp_logits, max=self.clamp_logits)
                        log_probs = self._get_answer_log_probs(next_query, action, logits)

                    if for_ref_model:
                        new_episode_steps.append({
                            **step,
                            'ref_log_probs': log_probs.detach().cpu(),
                        })
                    else:
                        new_episode_steps.append({
                            **step,
                            'log_probs': log_probs.detach().cpu(),
                        })
                new_trajectories.append({
                    **episode,
                    'steps': new_episode_steps,
                })

        return new_trajectories

    def _get_answer_log_probs(self, next_query: TokenizedDict, action: TokenizedDict, logits: torch.Tensor) -> torch.Tensor:
        query_lens = next_query['attention_mask'].sum(dim=1).long()  # Query lengths per sample
        answer_mask = action['attention_mask']
        answer_lens = answer_mask.sum(dim=1).long()  # Answer lengths per sample (before padding)
        max_length = next_query['input_ids'].size(1)

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
        shifted_targets = action['input_ids'][:, 1:] # Remove first answer token - deterministic [A] token
        shifted_mask = answer_mask[:, 1:] # Remove also first position from attention mask

        # 4. Calculate and mask new shifted log probs
        new_log_probs = F.log_softmax(shifted_logits, dim=-1)
        log_probs = new_log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
        log_probs *= shifted_mask

        return log_probs

    def train_epoch(self, dataloader: DataLoader, epoch: int, batch_size: int):
        """Train for one epoch."""
        # 1. Collect trajectories for current epoch
        self.actor.eval()
        trajectories = self.collect_trajectories(dataloader, epoch, batch_size)

        trajectories = self.get_pre_update_log_probs(trajectories, for_ref_model=True)

        if self.log_probs_source == 'update':
            trajectories = self.get_pre_update_log_probs(trajectories)

        self.actor.train()

        # 2. Flatten trajectories, call critic and collect values, dones and rewards, and calculate advantages
        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                values, rewards, dones = self._critic_values_rewards_and_dones(trajectories)
                advantages, ref_values = self.rl_algorithm.calculate_advantages(rewards, values, dones)
        else:
            values, rewards, dones = self._critic_values_rewards_and_dones(trajectories)
            advantages, ref_values = self.rl_algorithm.calculate_advantages(rewards, values, dones)

        # 3. Run internal update epochs
        critic_loss_sum, policy_loss_sum = 0.0, 0.0
        for update_epoch in range(self.update_epochs):
            # 4. Run 'on update epoch start' callbacks
            for cb in self.callbacks:
                cb.on_update_epoch_start(self.actor, self.critic, epoch, update_epoch)

            # 5. Run RL algorithm step
            policy_loss, critic_loss = self.rl_step(trajectories[:-1], advantages, ref_values, epoch, batch_size)

            if self.use_ddp:
                policy_loss = distributed_mean(policy_loss)
                critic_loss = distributed_mean(critic_loss)

            # 6. Run 'on update epoch end' callbacks
            for cb in self.callbacks:
                cb.on_update_epoch_end(self.actor, self.critic, epoch, update_epoch, policy_loss, critic_loss)

            # 7. Accumulate losses for epoch callbacks
            critic_loss_sum += critic_loss
            policy_loss_sum += policy_loss

        if self.replay_buffer is not None:
            print('Updating replay buffer')
            self.replay_buffer(trajectories)

        # 8. Return policy and critic mean losses for epoch callbacks
        return policy_loss_sum / self.update_epochs, critic_loss_sum / self.update_epochs

    def _eval_loader(self, batch_size: int):
        if self.use_ddp:
            return DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                pin_memory=True,
                sampler=DistributedSampler(self.eval_dataset, shuffle=False),
                collate_fn=MrlCurriculumDataset.collate_mrl_batch,
            )
        else:
            return DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=MrlCurriculumDataset.collate_mrl_batch,
            )

    def _eval_writer(self, avg_reward: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar('Eval/episode reward (global)', avg_reward, self.global_step['eval'])
            self.writer.add_scalar(f'Eval/episode reward (steps: {self.curriculum_steps}, epoch: {epoch})', avg_reward,
                                   self.epoch_step['eval'])
            self.writer.add_scalar(f'Eval/episode reward (steps: {self.curriculum_steps})', avg_reward,
                                   self.stage_step['eval'])

    def evaluate(self, batch_size: int, epoch: int):
        """Evaluate model on validation dataset."""
        # 1. Init evaluation DataLoader
        dataloader = self._eval_loader(batch_size)
        total_reward = torch.tensor(0.0).to(self.device)
        count = torch.tensor(0).to(self.device)

        self.actor.eval()

        # 2. Run evaluation on all batch episodes
        for batch in dataloader:
            with torch.no_grad():
                if batch['query']['input_ids'].size(0) == batch_size:
                    self._increment_steps('eval')
                    # 3. Reset STM with random resets ratio and reward model running mean
                    reset_stm = self.reset_stm()
                    self.reward.reset_running_mean()

                    # 4. Get batches for first queries, answers and all follow-up interactions
                    first_query, first_answer, interactions = batch['query'], batch['answer'], batch['interactions']
                    # 5. Encode and update STM with initial interactions (batch)
                    first_interaction = self._move_multiple_batches(first_query, first_answer)
                    if reset_stm:
                        self.memory_warmup(*first_interaction)
                    self.encode_and_update_stm(*first_interaction)

                    # 6. Save follow-up interactions len and first query and answer as previous one for iteration
                    interactions_len = len(interactions)
                    query, answer = first_query, first_answer
                    episode_reward = torch.tensor(0.0).to(self.device)
                    episode_interactions = torch.tensor(0).to(self.device)

                    prev_interaction = None

                    # 7. Run all follow-up interactions
                    for i, interaction in enumerate(interactions):
                        # 8. Generate batch of answers
                        next_query = self._move_batch(interaction['query'])
                        generated_answer, _ = self.generate_answer(next_query)

                        is_last_interaction = (i + 1) == interactions_len

                        detached_answer = self._batch_detach(generated_answer)

                        # 9. Depending on current strategy and step, compute reward
                        if self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0:
                            reward = self.compute_reward(
                                detached_answer, interaction['answer'], (query, answer),
                                interaction['query'], mode=MrlRewardMode.NEGATIVE, eval_mode=True
                            )
                        elif self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and is_last_interaction:
                            reward = self.compute_reward(
                                detached_answer, interaction['answer'], (first_query, first_answer),
                                interaction['query'], mode=MrlRewardMode.LONG_RANGE, eval_mode=True,
                                prev_data=prev_interaction,
                            )
                        else:
                            reward = self.compute_reward(
                                detached_answer, interaction['answer'], (query, answer),
                                interaction['query'], mode=MrlRewardMode.STANDARD, eval_mode=True,
                                prev_data=prev_interaction
                            )

                        # 10. Encode and update memory for the next interaction
                        if not is_last_interaction:
                            self.encode_and_update_stm(
                                next_query,
                                self._move_batch(interaction['answer']) if self.teacher_forcing else generated_answer
                            )

                        # 11. Accumulate rewards
                        step_reward = reward.mean()
                        # total
                        total_reward += step_reward
                        count += 1
                        # episode
                        episode_reward += step_reward
                        episode_interactions += 1
                        # 12. Save previous interaction
                        if not (self.strategy == MrlStrategy.LONG_RANGE_STRATEGY and i == 0):
                            prev_interaction = (query, answer)
                        query, answer = interaction['query'], (interaction['answer'] if self.teacher_forcing else self._cpu_detach(generated_answer))
                    avg_episode_reward = (episode_reward / episode_interactions).item()
                    # 13. Run eval TensorBoard writer with average episode reward
                    self._eval_writer(avg_episode_reward, epoch)

                    # 14. Run "on eval episode end" callbacks
                    for cb in self.callbacks:
                        cb.on_eval_episode_end(self.actor, epoch, self.epoch_step['eval'], avg_episode_reward)

        # 15. Calculate average reward
        avg_reward = (total_reward / count) if count > 0 else torch.tensor(0.0).to(self.device)
        if self.use_ddp:
            avg_reward = distributed_mean(avg_reward)

        avg_reward = avg_reward.item()

        should_stop_stage = False
        # 16. Run "on eval end" callbacks
        for cb in self.callbacks:
            should_stop = cb.on_eval_end(self.actor, self.critic, epoch, avg_reward)
            if should_stop:
                should_stop_stage = True

        self.actor.train()

        return should_stop_stage

    def _apply_unfreeze_strategy(self, epoch: int, unfreeze_epoch: UnfreezeEpochsStrategy):
        is_scheduled_unfreeze = isinstance(unfreeze_epoch, tuple)
        if is_scheduled_unfreeze:
            fetch_epoch, update_epoch, all_epoch = unfreeze_epoch

            if isinstance(fetch_epoch, tuple):
                switch_epoch, unfreeze_lr = fetch_epoch
                if epoch == switch_epoch:
                    self.actor.unfreeze_components(freeze_embeddings=self.freeze_embeddings)
                    self.optimizer = self._init_unfreeze_optimizer('fetch', unfreeze_lr)
                    print(f"Activating 'fetch' unfreeze strategy with custom decoder/encoder lr: {unfreeze_lr}")
            elif epoch == fetch_epoch:
                self.actor.freeze_components('fetch', freeze_embeddings=self.freeze_embeddings)
                print(
                    f"Activating 'fetch' unfreeze strategy - memory-attention and decoder cross-attention trainable / rest of the model frozen")
            if isinstance(update_epoch, tuple):
                switch_epoch, unfreeze_lr = update_epoch
                if epoch == switch_epoch:
                    self.actor.unfreeze_components(freeze_embeddings=self.freeze_embeddings)
                    self.optimizer = self._init_unfreeze_optimizer('update', unfreeze_lr)
                    print(f"Activating 'update' unfreeze strategy with custom decoder not memory lr: {unfreeze_lr}")
            elif epoch == update_epoch:
                self.actor.freeze_components('update', freeze_embeddings=self.freeze_embeddings)
                print(
                    f"Activating 'update' unfreeze strategy - memory-attention, encoder and decoder cross-attention trainable / rest of decoder frozen")
            if epoch == all_epoch:
                self.actor.unfreeze_components(freeze_embeddings=self.freeze_embeddings)
                self.optimizer = self._init_unfreeze_optimizer('all', 0.)
                print(f"Switching to train 'all' strategy - unfreeze all components")
        elif epoch == unfreeze_epoch:
            self.actor.unfreeze_components(freeze_embeddings=self.freeze_embeddings)
            print(f"Switching to train 'all' strategy - unfreeze all components")

    def _init_unfreeze_optimizer(
            self,
            mode: Literal['update', 'fetch', 'all'],
            unfreeze_lr: float,
    ) -> torch.optim.Optimizer:
        memory_lr = self.optim_config['memory_lr'] if 'memory_lr' in self.optim_config else self.optim_config['lr']
        memory_attn_lr = self.optim_config['memory_attn_lr'] if 'memory_attn_lr' in self.optim_config else self.optim_config['lr']
        model_lr, embedding_lr, encoder_lr = self.optim_config['lr'], self.optim_config['embedding_lr'], self.optim_config['encoder_lr']

        if mode == 'update':
            params = [
                {'params': self.actor.embedding_parameters(), 'lr': embedding_lr},
                {'params': self.actor.encoder.not_memory_parameters(), 'lr': encoder_lr},
                {'params': self.actor.memory_attention_parameters(), 'lr': memory_attn_lr},
                {'params': self.actor.decoder.memory_parameters(), 'lr': memory_lr},
                {'params': self.actor.decoder.not_memory_parameters(), 'lr': unfreeze_lr},
            ]
        elif mode == 'fetch':
            params = [
                {'params': self.actor.embedding_parameters(), 'lr': embedding_lr},
                {'params': self.actor.encoder.not_memory_parameters(), 'lr': unfreeze_lr},
                {'params': self.actor.memory_attention_parameters(), 'lr': memory_attn_lr},
                {'params': self.actor.decoder.memory_parameters(), 'lr': memory_lr},
                {'params': self.actor.decoder.not_memory_parameters(), 'lr': unfreeze_lr},
            ]
        else:
            params = [
                {'params': self.actor.embedding_parameters(), 'lr': embedding_lr},
                {'params': self.actor.encoder.not_memory_parameters(), 'lr': encoder_lr},
                {'params': self.actor.memory_attention_parameters(), 'lr': memory_attn_lr},
                {'params': self.actor.decoder.memory_parameters(), 'lr': memory_lr},
                {'params': self.actor.decoder.not_memory_parameters(), 'lr': model_lr},
            ]

        return torch.optim.AdamW(params, weight_decay=self.optim_config['weight_decay'])

    def _setup_curriculum_step(self, config: CurriculumConfig) -> tuple[
        tuple[int, UnfreezeEpochsStrategy], tuple[bool, int, float]]:
        # 1. Set common fields based on config
        self.curriculum_steps = config.get('steps', 1)  # number of steps to run in episode
        self.train_dataset = config.get('dataset', None)  # training dataset for current curriculum stage
        self.eval_dataset = config.get('eval_dataset', None)  # evaluation dataset for current curriculum stage
        self.callbacks = config.get('callbacks',
                                    self.shared_callbacks)  # trainer callbacks for current curriculum stage
        self.strategy = config.get('strategy',
                                   MrlStrategy.MULTI_STEP_STRATEGY)  # MRL strategy for given curriculum stage
        self.reward = config.get('reward_model', self.shared_reward_model)  # MRL Reward Model for curriculum stage
        self.update_epochs = config.get('update_epochs', self.shared_update_epochs)  # Internal update epochs
        self.freeze_embeddings = config.get('freeze_embeddings', self.shared_freeze_embeddings)
        self.teacher_forcing = config.get('teacher_forcing', False)

        self.use_memory_warmup = config.get('use_memory_warmup', self.base_memory_warmup_config['use_memory_warmup'])
        self.hard_warmup = config.get('hard_warmup', self.base_memory_warmup_config['hard_warmup'])

        def has_param(field: OptimField) -> bool:
            return field in config and config[field] is not None

        optim_params: list[OptimField] = ['lr', 'critic_lr', 'weight_decay', 'critic_weight_decay', 'encoder_lr']
        mem_optim_params: list[OptimField] = ['memory_lr', 'memory_attn_lr']

        has_any_optim_param = any(
            has_param(field) for field in optim_params
        ) or (has_param('separate_memory_lr') and config['separate_memory_lr'] and any(
            has_param(field) for field in mem_optim_params
        ))

        if has_any_optim_param:
            if config.get('separate_memory_lr', False):
                self.optim_config = {
                    'lr': config.get('lr', self.base_optim_config['lr']),
                    'critic_lr': config.get('critic_lr', self.base_optim_config['critic_lr']),
                    'weight_decay': config.get('weight_decay', self.base_optim_config['weight_decay']),
                    'critic_weight_decay': config.get('critic_weight_decay',
                                                      self.base_optim_config['critic_weight_decay']),
                    'critic_encoder_lr': config.get('critic_encoder_lr', self.base_optim_config['critic_encoder_lr']),
                    'memory_lr': config.get('memory_lr', self.base_optim_config['memory_lr']),
                    'embedding_lr': config.get('embedding_lr', self.base_optim_config['embedding_lr']),
                    'encoder_lr': config.get('encoder_lr', self.base_optim_config['encoder_lr']),
                    'memory_attn_lr': config.get('memory_attn_lr', self.base_optim_config['memory_attn_lr']),
                }
            else:
                self.optim_config = {
                    'lr': config.get('lr', self.base_optim_config['lr']),
                    'critic_lr': config.get('critic_lr', self.base_optim_config['critic_lr']),
                    'weight_decay': config.get('weight_decay', self.base_optim_config['weight_decay']),
                    'critic_weight_decay': config.get('critic_weight_decay',
                                                      self.base_optim_config['critic_weight_decay']),
                    'critic_encoder_lr': config.get('critic_encoder_lr', self.base_optim_config['critic_encoder_lr']),
                    'embedding_lr': config.get('embedding_lr', self.base_optim_config['embedding_lr']),
                    'encoder_lr': config.get('encoder_lr', self.base_optim_config['encoder_lr']),
                }
            self.optimizer, self.critic_optimizer = self._init_optimizers(**self.optim_config)
        elif self.optim_config != self.base_optim_config:
            self.optim_config = self.base_optim_config
            self.optimizer, self.critic_optimizer = self._init_optimizers(**self.optim_config)

        if 'replay_buffer' in config and config['replay_buffer'] is not None:
            self.replay_buffer = config['replay_buffer']
        else:
            self.replay_buffer = None

        # 2. Get epochs and random resets configs
        epochs = config.get('epochs', 5)  # number of epochs for current stage
        unfreeze_epoch = config.get('unfreeze_epoch',
                                    0)  # epoch when components (other than memory) are unfrozen (before epoch starts)
        random_resets = config.get('random_resets',
                                   False)  # flag for using random STM resets (recommended, as model should learn transitions between different states)
        random_resets_from = config.get('random_resets_from', None)  # epoch from which random STM resets are started
        random_resets_ratio = config.get('random_resets_ratio',
                                         None)  # ratio of random STM resets - 1.0 is "always reset", 0.0 is "no resets"

        # 3. Reset stage step counter
        self.stage_step = self._init_steps()

        return (epochs, unfreeze_epoch), (random_resets, random_resets_from, random_resets_ratio)

    def __call__(self, curriculum_config: list[CurriculumConfig], batch_size: int, ddp_find_unused_parameters: bool = False):
        """Start Memory Reinforcement Learning Curriculum."""

        # 0. Set global epoch count for all stages
        self.global_epochs_count = sum(stage['epochs'] for stage in curriculum_config)
        self.global_epoch = 0

        # 1. Init DDP for distributed training mode
        if self.use_ddp:
            rank, world_size = get_os_ddp_config()
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            self.actor = DistributedDataParallel(self.actor, device_ids=[self.device.index], find_unused_parameters=ddp_find_unused_parameters)
            self.critic = DistributedDataParallel(self.critic, device_ids=[self.device.index])
            self.reference_model = DistributedDataParallel(self.reference_model, device_ids=[self.device.index])

        self.reference_model = self.reference_model.to(self.device)
        # 2. Init BatchSampler with actor model (we have to run it after DDP init)
        self.generator = BatchSampler(self.actor, self.device, end_token_id=self.end_token_id, answer_token_id=self.answer_token_id, pad_token_id=self.pad_token_id, use_self_attn_cache=self.use_self_attn_cache)

        # 3. Run each curriculum step based on config
        for current_curriculum_step in curriculum_config:
            # 4. Setup training config for curriculum step
            epochs_config, random_resets_config = self._setup_curriculum_step(current_curriculum_step)
            epochs, unfreeze_epoch = epochs_config
            random_resets, random_resets_from, random_resets_ratio = random_resets_config
            assert self.train_dataset is not None

            # 5. Freeze all components except memory attention and memory cross-attention layers in decoder/encoder
            if unfreeze_epoch != 0:
                if callable(unfreeze_epoch):
                    unfreeze_epoch(-1, self)
                elif isinstance(unfreeze_epoch, tuple):
                    self.actor.freeze_components('warmup', freeze_embeddings=self.freeze_embeddings)
                    print(
                        f"Starting training with complex unfreeze schedule - 'warmup' - only memory-attention trainable / rest model frozen"
                    )
                else:
                    self.actor.freeze_components('fetch', freeze_embeddings=self.freeze_embeddings)
                    print(
                        f"Starting training with simple unfreeze schedule - 'fetch' - memory-attention, encoder and decoder's cross-attention trainable / rest model frozen"
                    )

            # 6. Setup train DataLoader
            if self.use_ddp:
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    pin_memory=True,
                    collate_fn=MrlCurriculumDataset.collate_mrl_batch,
                    drop_last=True,
                )
            else:
                train_sampler = None
                dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    collate_fn=MrlCurriculumDataset.collate_mrl_batch,
                )

            self.critic.train()

            # 7. Run selected number of epochs for given curriculum stage
            for epoch in range(epochs):
                # 8. Increment global epoch
                self.global_epoch += 1
                # 9. Run "on epoch start" callbacks (log info, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_start(self.actor, epoch, epochs, current_curriculum_step, self.global_epoch,
                                      self.global_epochs_count)

                # 10. Reset steps counter for epoch
                self.epoch_step = self._init_steps()

                # 11. Set random STM resets ratio from selected epoch
                if random_resets and random_resets_from <= epoch:
                    self.random_resets_ratio = random_resets_ratio
                else:
                    self.random_resets_ratio = 1.0

                # 12. Apply the unfreeze strategy
                if callable(unfreeze_epoch):
                    unfreeze_epoch(epoch, self)
                else:
                    self._apply_unfreeze_strategy(epoch, unfreeze_epoch)

                # 13. Set epoch for distributed sampler
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                # 14. Run reinforcement learning algorithms for current epoch
                policy_loss, critic_loss = self.train_epoch(dataloader, epoch, batch_size)

                # 15. If evaluation dataset is provided, run evaluation steps
                if self.eval_dataset:
                    should_stop_stage = self.evaluate(batch_size, epoch)
                else:
                    should_stop_stage = False

                # 16. Finally, run "on epoch end" callbacks (save models, etc.)
                for cb in self.callbacks:
                    cb.on_epoch_end(self.actor, epoch, epochs, policy_loss, critic_loss, self.global_epoch,
                                    self.global_epochs_count)

                # 17. Synchronize TensorBoard writer
                if self.writer:
                    self.writer.flush()

                # 18. Synchronize devices in DDP mode
                if self.use_ddp:
                    dist.barrier()

                # 19. Finish curriculum stage if rewards are not increased or reached threshold point
                if should_stop_stage:
                    break

            # 20. Run "on_training_end" callbacks after each curriculum stage (they have own callbacks)
            for cb in self.callbacks:
                cb.on_training_end(self.actor, self.critic, current_curriculum_step)

        self.actor.eval()

        # 21. Training end - finish processes after all curriculum stages
        if self.use_ddp:
            dist.destroy_process_group()

        # 22. Close writer
        if self.writer:
            self.writer.close()
