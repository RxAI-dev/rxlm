import torch
import torch.nn as nn
from typing import TypedDict, Iterator


class TokenizedDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

def smart_concat_critic_states(
        prev_query: TokenizedDict,
        prev_answer: TokenizedDict,
        next_query: TokenizedDict,
        max_length: int,
        pad_token_id: int
) -> TokenizedDict:
    """
    Smart vectorized concatenation of MRL critic states - previous interaction (query and answer) and next query.
    It creates a batch of critic input sequences from previous query, previous answer and next query batches.
    Used in MRL to concatenate critic states in correct format.

    All the concatenated sequences (batches) are padded to the same max length, but the result should have two times
    longer max length. Single max length is made to fit single query and answer, but here we have additional next query,
    so we are using 2x longer sequence for safety.

    Args:
        prev_query (TokenizedDict): Batch of tokenized queries with attention masks from previous interaction
        prev_answer (TokenizedDict): Batch of tokenized answers with attention masks from previous interaction
        next_query (TokenizedDict): Batch of tokenized queries with attention masks from next interaction
        max_length (int): Max length of result sequence.
        pad_token_id (int): Index of padding token
    """
    device = prev_query['input_ids'].device
    batch_size = prev_query['input_ids'].size(0)

    # Get input dimensions
    query_max_len = prev_query['input_ids'].size(1)
    answer_max_len = prev_answer['input_ids'].size(1)
    next_q_max_len = next_query['input_ids'].size(1)

    # Get actual lengths using attention masks
    query_lens = prev_query['attention_mask'].sum(dim=1)
    answer_lens = prev_answer['attention_mask'].sum(dim=1)
    next_query_lens = next_query['attention_mask'].sum(dim=1)

    # Calculate positions and boundaries
    positions = torch.arange(max_length, device=device).expand(batch_size, -1)
    section1_end = query_lens.unsqueeze(1)
    section2_end = section1_end + answer_lens.unsqueeze(1)
    section3_end = section2_end + next_query_lens.unsqueeze(1)

    # Create masks for each section
    mask_prev = positions < section1_end
    mask_answer = (positions >= section1_end) & (positions < section2_end)
    mask_next = (positions >= section2_end) & (positions < section3_end)

    # Build combined tensor
    combined_ids = torch.full((batch_size, max_length), pad_token_id, device=device)

    # 1. Fill previous query section (with input length clamping)
    query_indices = positions.clamp(max=query_max_len - 1)
    combined_ids = torch.where(
        mask_prev,
        prev_query['input_ids'].gather(1, query_indices),
        combined_ids
    )

    # 2. Fill answer section (with answer length clamping)
    answer_pos = (positions - section1_end).clamp(min=0, max=answer_max_len - 1)
    combined_ids = torch.where(
        mask_answer,
        prev_answer['input_ids'].gather(1, answer_pos),
        combined_ids
    )

    # 3. Fill next query section (with next query length clamping)
    next_q_pos = (positions - section2_end).clamp(min=0, max=next_q_max_len - 1)
    combined_ids = torch.where(
        mask_next,
        next_query['input_ids'].gather(1, next_q_pos),
        combined_ids
    )

    # Create attention mask
    combined_mask = (positions < section3_end).long()

    return {
        'input_ids': combined_ids,
        'attention_mask': combined_mask
    }

def smart_concat(query: TokenizedDict, answer: TokenizedDict, max_length: int, pad_token_id: int) -> TokenizedDict:
    """
    Smart vectorized concatenation of interaction parts - query and answer. It creates
    batch of interactions from query and answer batches. Used in MRL to concatenate data
    to encode and update memory.

    Query and answer sequences are padded to the same max length, and the result also has
    the same length.

    Args:
        query (TokenizedDict): Batch of tokenized queries with attention masks
        answer (TokenizedDict): Batch of tokenized answers with attention masks
        max_length (int): Max length of each sequence - query, answer and result.
        pad_token_id (int): Index of padding token
    """
    device = query['input_ids'].device
    batch_size = query['input_ids'].size(0)

    # Get actual lengths from attention masks
    query_lens = query['attention_mask'].sum(dim=1)
    answer_lens = answer['attention_mask'].sum(dim=1)

    # Create combined length tensor
    combined_lens = torch.minimum(query_lens + answer_lens,
                                  torch.full_like(query_lens, max_length))

    # Create position indices [batch_size, max_length]
    positions = torch.arange(max_length, device=device).expand(batch_size, -1)

    # Create mask for query/answer parts
    query_mask = positions < query_lens.unsqueeze(1)
    answer_mask = (positions >= query_lens.unsqueeze(1)) & (positions < combined_lens.unsqueeze(1))

    # Calculate answer positions with overflow protection
    answer_pos = (positions - query_lens.unsqueeze(1)).clamp(min=0)

    # Build combined_ids using vectorized where
    combined_ids = torch.where(
        query_mask,
        query['input_ids'].gather(1, torch.minimum(positions, query_lens.unsqueeze(1) - 1).long()),
        torch.where(
            answer_mask,
            answer['input_ids'].gather(1, answer_pos),
            query['input_ids'].new_full((1,), pad_token_id)
        )
    )

    # Build attention mask
    combined_mask = (positions < combined_lens.unsqueeze(1)).long()

    return {
        'input_ids': combined_ids,
        'attention_mask': combined_mask
    }

def get_gradient_norms(params: Iterator[nn.Parameter]):
    total_norm = 0
    grad_params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    for p in grad_params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    params_len = len(grad_params)
    if params_len != 0:
        mean_norm = total_norm / params_len
    else:
        mean_norm = 0.0
    return total_norm, mean_norm
