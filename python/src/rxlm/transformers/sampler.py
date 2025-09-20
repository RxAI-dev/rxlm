import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, Union, Optional
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer


def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Apply top-p (nucleus) sampling
    if top_p is not None and 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted indices back to original positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample from distribution
    return torch.multinomial(probs, num_samples=1)


class Sampler:
    def __init__(self, model: nn.Module, device: torch.device, end_token_id: int):
        self.model = model.to(device)
        self.device = device
        self.end_token_id = end_token_id

    def _generate_token(
            self,
            input_ids: torch.Tensor,
            temperature: float,
            top_k: int,
            top_p: float,
            attention_mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        # Forward pass to get next token logits
        outputs = self.model(input_ids, attention_mask=attention_mask)
        next_token_logits = outputs[:, -1, :]  # Get logits for next token
        # Apply sampling
        next_token = sample(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        next_token = next_token.item()  # Extract scalar token
        next_token_ten = torch.tensor([[next_token]], device=self.device)
        next_input_ids = torch.cat([input_ids, next_token_ten], dim=1)
        new_one = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        next_mask = torch.cat([attention_mask, new_one], dim=1) if attention_mask is not None else None
        # Yield the generated token
        return (
            next_token,
            next_input_ids,
            next_mask
        )

    def __call__(
            self,
            initial_tokens: torch.Tensor,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_seq_len: int = 50,
            attention_mask: torch.Tensor = None,
            no_grad: bool = True,
    ) -> Iterator[int]:
        # Convert initial tokens to tensor and move to device
        input_ids = initial_tokens

        if no_grad:
            with torch.no_grad():
                for _ in range(max_seq_len):
                    next_token, input_ids, attention_mask = self._generate_token(input_ids, temperature, top_k, top_p,
                                                                                 attention_mask)
                    yield next_token
                    if next_token == self.end_token_id:
                        break
        else:
            for _ in range(max_seq_len):
                next_token, input_ids, attention_mask = self._generate_token(input_ids, temperature, top_k, top_p,
                                                                             attention_mask)
                yield next_token
                if next_token == self.end_token_id:
                    break


class SampleDecoder:
    def __init__(self, sampler: Sampler, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = self.sampler.device

    def tokenize_input(self, text: str, max_seq_len: int = 256):
        tokenized = self.tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors='pt',
            return_attention_mask=True
        )
        tokenized['input_ids'] = tokenized['input_ids'][:, :-1].to(self.device)
        tokenized['attention_mask'] = tokenized['attention_mask'][:, :-1].to(self.device)
        del tokenized['token_type_ids']
        return tokenized

    def ids_iter(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        tokenized = self.tokenize_input(text, max_seq_len=max_seq_len)
        return self.sampler(
            tokenized['input_ids'],
            temperature=temperature,
            top_p=top_p,
            max_seq_len=max_seq_len,
            attention_mask=tokenized['attention_mask']
        )

    def txt_iter(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        return map(
            lambda x: self.tokenizer.decode([x]).replace('Ċ', '\n').replace('Ġ', ' '),
            self.ids_iter(text, temperature, top_p, max_seq_len)
        )

    def txt(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        return text + ''.join(self.txt_iter(text, temperature, top_p, max_seq_len))

    def print_stream(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len=256):
        print(text, end='')
        resp = text
        for txt_token in self.txt_iter(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len):
            print(txt_token, end='')
            resp += txt_token
        return resp

    def __call__(self, text: str, print_stream: bool = False, temperature: float = 0.1, top_p: float = 0.9,
                 max_seq_len=256):
        if print_stream:
            return self.print_stream(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len)
        else:
            return self.txt(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len)

class InteractionSampler(SampleDecoder):
    def __init__(self, sampler: Sampler, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        super(InteractionSampler, self).__init__(sampler, tokenizer)

    def txt(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len: int = 256, special_token_spaces: bool = True):
        txt = '[Q]' + text + '[A]'
        start_txt = '[Q] ' + text + ' [A] ' if special_token_spaces else txt
        return start_txt + ''.join(self.txt_iter(txt, temperature, top_p, max_seq_len))

    def print_stream(self, text: str, temperature: float = 0.1, top_p: float = 0.9, max_seq_len: int = 256, special_token_spaces: bool = True):
        txt = '[Q]' + text + '[A]'
        start_txt = '[Q] ' + text + ' [A] ' if special_token_spaces else txt
        print(start_txt, end='')
        resp = start_txt
        for txt_token in self.txt_iter(txt, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len):
            print(txt_token, end='')
            resp += txt_token
        return resp

    def __call__(self, text: str, print_stream: bool = False, temperature: float = 0.1, top_p: float = 0.9,
                 max_seq_len: int = 256, special_token_spaces: bool = True):
        if print_stream:
            return self.print_stream(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, special_token_spaces=special_token_spaces)
        else:
            return self.txt(text, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, special_token_spaces=special_token_spaces)


def sample_batch(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sampled_tokens, log_probs)"""
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    # Store original device
    device = logits.device

    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, torch.tensor(-float('inf'), device=device), logits)

    # Apply top-p filtering
    if top_p is not None and 0 < top_p <= 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Calculate cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask to filter tokens
        sorted_mask = cumulative_probs <= top_p
        sorted_mask[..., 0] = True  # Ensure at least one token is kept

        # Create mask for original indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)

        # Apply mask
        logits = torch.where(mask, logits, torch.tensor(-float('inf'), device=device))

    # At this point ensure at least one token is available per batch element
    alive = torch.isfinite(logits).any(dim=-1)
    if not alive.all():
        # Force keep the most probable token for dead rows
        max_indices = logits.argmax(dim=-1)
        logits[~alive] = -float('inf')
        logits.scatter_(dim=-1, index=max_indices.unsqueeze(-1), value=0)

    # Calculate log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Convert to probabilities
    probs = torch.exp(log_probs)

    # Ensure numerical stability for sampling
    probs = probs.clamp(min=1e-12)

    # Sample tokens
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Gather log probabilities
    selected_log_probs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    # Convert back to original dtype
    return next_tokens.long(), selected_log_probs


class BatchSampler:
    def __init__(self, model: nn.Module, device: torch.device, end_token_id: int, answer_token_id: int, use_self_attn_cache: bool = True, first_normal_token_id: int = None, pad_token_id: int = 0):
        self.model = model.to(device)
        self.device = device
        self.end_token_id = end_token_id
        self.answer_token_id = answer_token_id
        self.pad_token_id = pad_token_id
        self.use_self_attn_cache = use_self_attn_cache
        self.first_normal_token_id = first_normal_token_id if first_normal_token_id is not None else (self.answer_token_id + 1)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_seq_len = input_ids.shape

        initial_lens = attention_mask.sum(dim=1)
        for i in range(batch_size):
            input_ids[i, initial_lens[i]] = self.answer_token_id
            attention_mask[i, initial_lens[i]] = 1

        initial_lens += 1
        current_lens = initial_lens.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        log_probs = torch.zeros((batch_size, max_gen_len), dtype=dtype, device=self.device)
        working_ids = input_ids.clone()
        working_mask = attention_mask.clone()

        with torch.set_grad_enabled(not no_grad):
            stm_kv_cache = self.model.prepare_stm_kv_cache()

        self.model.reset_self_attn_cache()

        for step in range(max_gen_len):
            active = (~finished) & (current_lens < max_seq_len)
            if not active.any():
                break

            max_len = current_lens.max()

            with torch.set_grad_enabled(not no_grad):
                indices = (current_lens - 1).to(self.device)
                # Slice input and mask up to the current max length among active sequences
                if step == 0 or not self.use_self_attn_cache:
                    # prompt phase - cache input sequence
                    inputs = working_ids[:, :max_len]
                    masks = working_mask[:, :max_len]

                    logits = self.model(
                        inputs,
                        attention_mask=masks,
                        stm_kv_cache=stm_kv_cache,
                        use_self_attn_cache=self.use_self_attn_cache,
                        current_positions=None,
                    )
                else:
                    # generate phase, use last token
                    finished_tokens = finished.unsqueeze(-1)
                    row_idx = torch.arange(working_ids.size(0))

                    selected_ids = working_ids[row_idx, indices].unsqueeze(-1)
                    selected_masks = working_mask[row_idx, indices].unsqueeze(-1)

                    inputs = torch.where(finished_tokens == 0, selected_ids, self.pad_token_id)
                    masks = torch.where(finished_tokens == 0, selected_masks, self.pad_token_id)

                    logits = self.model(
                        inputs,
                        attention_mask=masks,
                        stm_kv_cache=stm_kv_cache,
                        use_self_attn_cache=self.use_self_attn_cache,
                        current_positions=indices,
                    )

            # Get the last valid token index for each active sequence
            if step == 0 or not self.use_self_attn_cache:
                last_logits = logits[torch.arange(batch_size, device=self.device), indices]
            else:
                last_logits = logits[:, -1]

            # Sample next tokens and log probs
            next_tokens, step_log_probs = sample_batch(
                last_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Empty first token sampling workaround
            if step == 0:
                random_token = torch.randint(self.first_normal_token_id, logits.size(-1), size=(), device=self.device)
                random_log_prob = torch.log(torch.rand(size=(), device=self.device))

                next_tokens = torch.where(next_tokens == 0, random_token, next_tokens)
                step_log_probs = torch.where(next_tokens == 0, random_log_prob, step_log_probs)

            # Prepare active sequences mask
            active_mask = (~finished) & (current_lens < max_seq_len)
            if not active_mask.any():
                break
            # Get positions to update
            positions_to_update = current_lens
            # Prepare indexing batch range
            batch_range = torch.arange(batch_size, device=self.device)
            # Vectorized working tensors update
            working_ids[batch_range[active_mask], positions_to_update[active_mask]] = next_tokens[active_mask]
            working_mask[batch_range[active_mask], positions_to_update[active_mask]] = 1
            # Vectorized log probs update
            log_probs[batch_range[active_mask], step] = step_log_probs[active_mask].to(dtype=log_probs.dtype)
            # Update lens for active tokens
            current_lens += active_mask.long()
            # Update finished tensor if some batch items stopped generation
            finished |= (next_tokens == self.end_token_id) & active_mask

        # Extract generated tokens
        generated_ids = torch.zeros((batch_size, max_gen_len), dtype=torch.long, device=self.device)
        generated_mask = torch.zeros((batch_size, max_gen_len), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            start = initial_lens[i].item()
            end = current_lens[i].item()
            gen_len = min(end - start + 1, max_gen_len) # +1 for added [A] token
            if gen_len > 0:
                generated_ids[i, :gen_len] = working_ids[i, start-1:end] # -1 to include [A] token
                generated_mask[i, :gen_len] = working_mask[i, start-1:end] # -1 to include [A] token

        return generated_ids, generated_mask, log_probs


class BatchSampleDecoder:
    def __init__(
            self,
            sampler: BatchSampler,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            bos_token: str = '[BOS]',
            query_token: str = '[Q]',
    ):
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = self.sampler.device
        self.bos_token = bos_token
        self.query_token = query_token

    def tokenize_batch(self, texts: list[str], max_seq_len: int = 256):
        tokenized = self.tokenizer(
            [f'{self.bos_token}{self.query_token}{txt}' for txt in texts],
            max_length=max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=False
        )
        return {
            'input_ids': tokenized['input_ids'].to(self.device),
            'attention_mask': tokenized['attention_mask'].to(self.device)
        }

    def generate(
            self,
            texts: list[str],
            temperature: float = 1.0,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            max_seq_len: int = 256,
            no_grad: bool = True,
    ) -> list[str]:
        tokenized = self.tokenize_batch(texts, max_seq_len)
        generated_ids, _, _ = self.sampler(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_gen_len=max_seq_len,
            no_grad=no_grad,
        )

        decoded = []
        for seq in generated_ids:
            # Trim after end token
            end_pos = (seq == self.sampler.end_token_id).nonzero()
            if end_pos.size(0) > 0:
                seq = seq[:end_pos[0] + 1]
            decoded.append(self.tokenizer.decode(seq).replace('Ċ', '\n').replace('Ġ', ' '))

        return decoded

    def generate_with_log_probs(
            self,
            texts: list[str],
            temperature: float = 1.0,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            max_seq_len: int = 256,
            no_grad: bool = True,
    ) -> tuple[list[str], torch.Tensor]:
        tokenized = self.tokenize_batch(texts, max_seq_len)
        generated_ids, _, log_probs = self.sampler(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_gen_len=max_seq_len,
            no_grad=no_grad,
        )

        decoded = []
        for i, seq in enumerate(generated_ids):
            # Trim after end token
            end_pos = (seq == self.sampler.end_token_id).nonzero()
            if end_pos.size(0) > 0:
                seq = seq[:end_pos[0] + 1]
            decoded.append(self.tokenizer.decode(seq).replace('Ċ', '\n').replace('Ġ', ' '))

        return decoded, log_probs

    def __call__(
            self,
            texts: list[str],
            temperature: float = 1.0,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            max_seq_len: int = 256,
            no_grad: bool = True,
    ) -> list[str]:
        return self.generate(texts, temperature, top_p, top_k, max_seq_len, no_grad)