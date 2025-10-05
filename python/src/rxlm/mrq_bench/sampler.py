import torch
import torch.nn as nn
from typing import Optional
from datetime import datetime
from ..transformers.sampler import sample_batch


class MrqBenchBatchSampler:
    def __init__(
            self, model: nn.Module, device: torch.device, end_token_id: int, answer_token_id: int,
            use_self_attn_cache: bool = True, first_normal_token_id: int = None, pad_token_id: int = 0,
            use_stm_kv_cache: bool = True, use_first_empty_workaround: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.end_token_id = end_token_id
        self.answer_token_id = answer_token_id
        self.pad_token_id = pad_token_id
        self.use_self_attn_cache = use_self_attn_cache
        self.use_stm_kv_cache = use_stm_kv_cache
        self.first_normal_token_id = first_normal_token_id if first_normal_token_id is not None else (self.answer_token_id + 1)
        self.use_first_empty_workaround = use_first_empty_workaround

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_gen_len: int = 256,
        no_grad: bool = True,
        dtype: torch.dtype = torch.float32,
        timing_log: list[dict[str, float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_timing_mode = timing_log is not None

        init_time = datetime.timestamp(datetime.now())
        prompt_time = 0.0
        token_times = []

        batch_size, max_seq_len = input_ids.shape

        initial_lens = attention_mask.sum(dim=1)

        batch_range = torch.arange(batch_size, device=self.device)
        input_ids[batch_range, initial_lens] = self.answer_token_id
        attention_mask[batch_range, initial_lens] = 1
        initial_lens += 1

        current_lens = initial_lens.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        log_probs = torch.zeros((batch_size, max_gen_len), dtype=dtype, device=self.device)
        working_ids = input_ids.clone()
        working_mask = attention_mask.clone()

        if self.use_stm_kv_cache:
            with torch.set_grad_enabled(not no_grad):
                stm_kv_cache = self.model.prepare_stm_kv_cache()
        else:
            stm_kv_cache = None

        self.model.reset_self_attn_cache()

        for step in range(max_gen_len):
            active = (~finished) & (current_lens < max_seq_len)
            if not active.any():
                break

            max_len = current_lens.max()

            with torch.set_grad_enabled(not no_grad):
                indices = (current_lens - 1).to(self.device)
                if step == 0 or not self.use_self_attn_cache:
                    # prompt phase - cache input sequence
                    # Slice input and mask up to the current max length among active sequences
                    inputs = working_ids[:, :max_len]
                    masks = working_mask[:, :max_len]

                    if self.use_stm_kv_cache:
                        logits = self.model(
                            inputs,
                            attention_mask=masks,
                            stm_kv_cache=stm_kv_cache,
                            use_self_attn_cache=self.use_self_attn_cache,
                            current_positions=None,
                        )
                    else:
                        logits = self.model(
                            inputs,
                            attention_mask=masks,
                            use_self_attn_cache=self.use_self_attn_cache,
                            current_positions=None,
                        )
                else:
                    # generate phase, use last token
                    finished_tokens = finished.unsqueeze(-1)

                    selected_ids = working_ids[batch_range, indices].unsqueeze(-1)
                    selected_masks = working_mask[batch_range, indices].unsqueeze(-1)

                    inputs = torch.where(finished_tokens == 0, selected_ids, self.pad_token_id)
                    masks = torch.where(finished_tokens == 0, selected_masks, 0)

                    if self.use_stm_kv_cache:
                        logits = self.model(
                            inputs,
                            attention_mask=masks,
                            stm_kv_cache=stm_kv_cache,
                            use_self_attn_cache=self.use_self_attn_cache,
                            current_positions=indices,
                        )
                    else:
                        logits = self.model(
                            inputs,
                            attention_mask=masks,
                            use_self_attn_cache=self.use_self_attn_cache,
                            current_positions=indices,
                        )

            # Get the last valid token index for each active sequence
            if step == 0 or not self.use_self_attn_cache:
                last_logits = logits[batch_range, indices]
            else:
                last_logits = logits[:, -1]

            # Sample next tokens and log probs
            next_tokens, step_log_probs = sample_batch(
                last_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Empty first token sampling workaround
            if self.use_first_empty_workaround and step == 0:
                random_token = torch.randint(self.first_normal_token_id, logits.size(-1), size=(), device=self.device)
                random_log_prob = torch.log(torch.rand(size=(), device=self.device))

                next_tokens = torch.where(next_tokens == 0, random_token, next_tokens)
                step_log_probs = torch.where(next_tokens == 0, random_log_prob, step_log_probs)

            # Get positions to update
            positions_to_update = current_lens[active]
            active_range = batch_range[active]
            # Vectorized working tensors update
            working_ids[active_range, positions_to_update] = next_tokens[active]
            working_mask[active_range, positions_to_update] = 1
            # Vectorized log probs update
            log_probs[active_range, step] = step_log_probs[active].to(dtype=log_probs.dtype)
            # Update lens for active tokens
            current_lens += active.long()
            # Update finished tensor if some batch items stopped generation
            finished |= (next_tokens == self.end_token_id) & active

            if is_timing_mode:
                if step == 0:
                    prompt_time = datetime.timestamp(datetime.now())
                else:
                    token_times.append(datetime.timestamp(datetime.now()))

        # Extract generated tokens
        generated_ids = torch.zeros((batch_size, max_gen_len), dtype=torch.long, device=self.device)
        generated_mask = torch.zeros((batch_size, max_gen_len), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            start = initial_lens[i].item()
            end = current_lens[i].item()
            gen_len = min(end - start + 1, max_gen_len) # +1 for added [A] token
            if gen_len == max_gen_len:
                generated_ids[i, :gen_len] = working_ids[i, start-1:start-1+gen_len] # -1 to include [A] token
                generated_mask[i, :gen_len] = working_mask[i, start-1:start-1+gen_len] # -1 to include [A] token
            elif gen_len > 0:
                generated_ids[i, :gen_len] = working_ids[i, start-1:end] # -1 to include [A] token
                generated_mask[i, :gen_len] = working_mask[i, start-1:end] # -1 to include [A] token

        if is_timing_mode:
            token_timing = []
            token_init_time = prompt_time
            for token_time in token_times:
                token_timing.append(token_time - token_init_time)
                token_init_time = token_time

            timing = {
                'prompt': prompt_time - init_time,
                'token': sum(token_timing) / len(token_timing) if len(token_timing) > 0 else 0.0,
            }
            timing_log.append(timing)

        return generated_ids, generated_mask, log_probs
