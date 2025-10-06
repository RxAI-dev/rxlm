import torch
from typing import Optional, Literal


def tensor_sentence_bleu(
        candidates: torch.Tensor,
        references: torch.Tensor,
        pad_token_id: int = 0,
        max_n: int = 4,
        weights: Optional[torch.Tensor] = None,
        smoothing_method: Literal['none', 'floor', 'add-k', 'exp'] = 'exp',
        smoothing_epsilon: float = 0.1,
        smoothing_k: int = 1
) -> torch.Tensor:
    """
    Computes the BLEU score for each sentence in a batch.

    Args:
        candidates (torch.Tensor): Tensor token ids for candidate sentences. Shape: (batch_size, candidate_len).
        references (torch.Tensor): Tensor token ids for reference sentences. Shape: (batch_size, num_refs, ref_len).
        pad_token_id (int): Ignored padding token id (default 0).
        max_n (int): Maximum order of n-grams to include (default 4).
        weights (torch.Tensor): Weights for each row of n-grams. Tensor of dimension (max_n,).
        smoothing_method (str): Smoothing method. One from: ('none', 'floor', 'add-k', 'exp').
        smoothing_epsilon (float): Epsilon value for 'floor' method (default 0.1).
        smoothing_k (int): K value for 'add-k' method (default 1).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the BLEU score for each sentence.
    """
    if weights is None:
        weights = torch.ones(max_n, device=candidates.device) / max_n

    max_n = weights.size(0)
    batch_size = candidates.size(0)
    num_refs = references.size(1)

    candidate_mask = (candidates != pad_token_id)
    reference_mask = (references != pad_token_id)

    candidate_lengths = candidate_mask.sum(dim=1)
    reference_lengths = reference_mask.sum(dim=2)

    # Track numerators and denominators per sentence
    numerators = torch.zeros(batch_size, max_n, device=candidates.device, dtype=torch.long)
    denominators = torch.zeros(batch_size, max_n, device=candidates.device, dtype=torch.long)

    for n in range(1, max_n + 1):
        # Skip sentences that are too short for this n-gram order
        valid_for_n = (candidate_lengths >= n)
        if not torch.any(valid_for_n):
            continue

        # We only need to process sentences long enough for the current n-gram
        candidates_n = candidates[valid_for_n]
        references_n = references[valid_for_n]
        batch_size_n = candidates_n.size(0)

        candidate_ngrams = candidates_n.unfold(1, n, 1)
        candidate_mask_n = candidate_mask[valid_for_n]
        valid_candidate_mask = candidate_mask_n.unfold(1, n, 1).all(dim=2)
        candidate_ngrams_flat = candidate_ngrams[valid_candidate_mask]

        # Denominators are the number of n-grams in each candidate sentence
        denominators[valid_for_n, n - 1] = valid_candidate_mask.sum(dim=1)

        if candidate_ngrams_flat.size(0) == 0:
            continue

        reference_ngrams_list = []
        for i in range(num_refs):
            ref_i = references_n[:, i, :]
            ref_mask_i = reference_mask[valid_for_n, i, :]
            if torch.any(ref_mask_i.sum(dim=1) < n):
                # This part is complex to handle per-sentence, so we simplify
                # by unfolding and masking, which handles empty cases correctly.
                pass
            ref_ngrams_i = ref_i.unfold(1, n, 1)
            valid_ref_mask_i = ref_mask_i.unfold(1, n, 1).all(dim=2)
            reference_ngrams_list.append(ref_ngrams_i[valid_ref_mask_i])

        reference_ngrams_flat = torch.cat(reference_ngrams_list, dim=0)
        all_ngrams = torch.cat([candidate_ngrams_flat, reference_ngrams_flat], dim=0)

        if all_ngrams.size(0) == 0:
            continue

        unique_ngrams, inverse_indices = torch.unique(all_ngrams, dim=0, return_inverse=True)
        num_unique = unique_ngrams.size(0)

        cand_inv_indices = inverse_indices[:candidate_ngrams_flat.size(0)]
        ref_inv_indices = inverse_indices[candidate_ngrams_flat.size(0):]

        cand_batch_indices = torch.arange(batch_size_n, device=candidates.device).unsqueeze(1).expand_as(
            valid_candidate_mask)
        cand_batch_indices_flat = cand_batch_indices[valid_candidate_mask]
        cand_offsets = cand_batch_indices_flat * num_unique
        cand_bincount_indices = cand_inv_indices + cand_offsets
        candidate_counts = torch.bincount(
            cand_bincount_indices,
            minlength=batch_size_n * num_unique
        ).reshape(batch_size_n, num_unique)

        ref_max_counts = torch.zeros_like(candidate_counts)
        start_idx = 0
        for i in range(num_refs):
            ref_mask_i = reference_mask[valid_for_n, i, :]
            valid_ref_mask_i = ref_mask_i.unfold(1, n, 1).all(dim=2)
            num_ref_ngrams_i = valid_ref_mask_i.sum()

            if num_ref_ngrams_i == 0:
                continue

            ref_inv_indices_i = ref_inv_indices[start_idx: start_idx + num_ref_ngrams_i]
            start_idx += num_ref_ngrams_i

            ref_batch_indices = torch.arange(batch_size_n, device=candidates.device).unsqueeze(1).expand_as(
                valid_ref_mask_i)
            ref_batch_indices_flat = ref_batch_indices[valid_ref_mask_i]
            ref_offsets = ref_batch_indices_flat * num_unique
            ref_bincount_indices = ref_inv_indices_i + ref_offsets
            ref_counts_i = torch.bincount(
                ref_bincount_indices,
                minlength=batch_size_n * num_unique
            ).reshape(batch_size_n, num_unique)
            ref_max_counts = torch.maximum(ref_max_counts, ref_counts_i)

        clipped_counts = torch.minimum(candidate_counts, ref_max_counts)

        # Numerators are the sum of clipped counts for each sentence
        numerators[valid_for_n, n - 1] = clipped_counts.sum(dim=1)

    # All subsequent calculations are now element-wise for the batch
    precisions = torch.zeros(batch_size, max_n, device=candidates.device)
    if smoothing_method == 'floor':
        safe_numerators = torch.where(numerators == 0, smoothing_epsilon, numerators)
        safe_denominators = torch.where(denominators == 0, 1.0, denominators)
        precisions = safe_numerators / safe_denominators
    elif smoothing_method == 'add-k':
        precisions = (numerators + smoothing_k) / (denominators + smoothing_k)
    # The 'exp' smoothing method is tricky for per-sentence scores and less standard.
    # A simple approach is to fall back to a small value.
    else:  # 'none' or a simplified 'exp'
        mask = denominators > 0
        precisions[mask] = numerators[mask] / denominators[mask]
        # For 'exp' smoothing or to avoid log(0)
        if smoothing_method == 'exp':
            precisions[precisions == 0] = 1e-12

    closest_ref_lengths_indices = torch.abs(reference_lengths - candidate_lengths.unsqueeze(1)).argmin(dim=1)
    closest_ref_lengths = reference_lengths.gather(1, closest_ref_lengths_indices.unsqueeze(1)).squeeze(1)

    # Calculate brevity penalty per sentence
    brevity_penalty = torch.ones(batch_size, device=candidates.device)
    mask = candidate_lengths < closest_ref_lengths
    brevity_penalty[mask] = torch.exp(1 - closest_ref_lengths[mask] / candidate_lengths[mask])

    # Handle zero-length candidates
    brevity_penalty[candidate_lengths == 0] = 0.0

    # Final score calculation per sentence
    log_precisions = torch.log(precisions + 1e-12)  # Add epsilon to avoid log(0)
    weighted_log_precisions = weights.to(candidates.device) * log_precisions
    geo_mean = torch.exp(torch.sum(weighted_log_precisions, dim=1))

    score = brevity_penalty * geo_mean

    return score

def tensor_corpus_bleu(
    candidates: torch.Tensor,
    references: torch.Tensor,
    pad_token_id: int = 0,
    max_n: int = 4,
    weights: Optional[torch.Tensor] = None,
    smoothing_method: Literal['none', 'floor', 'add-k', 'exp'] = 'exp',
    smoothing_epsilon: float = 0.1,
    smoothing_k: int = 1
) -> torch.Tensor:
    """
    Computes the BLEU score for a batch of sequences in a fully vectorized and memory-efficient manner.
    This version avoids the out-of-memory problem by counting unique n-grams in a compact space instead of in a massive hash space.

    Args:
        candidates (torch.Tensor): Tensor token ids for candidate sentences. Shape: (batch_size, candidate_len).
        references (torch.Tensor): Tensor token ids for reference sentences. Shape: (batch_size, num_refs, ref_len).
        pad_token_id (int): Ignored padding token id (default 0).
        max_n (int): Maximum order of n-grams to include (default 4).
        weights (torch.Tensor): Weights for each row of n-grams. Tensor of dimension (max_n,).
        smoothing_method (str): Smoothing method. One from: ('none', 'floor', 'add-k', 'exp').
        smoothing_epsilon (float): Epsilon value for 'floor' method (default 0.1).
        smoothing_k (int): K value for 'add-k' method (default 1).

    Returns:
        torch.Tensor: A scalar tensor containing the BLEU score for the entire batch.
    """
    if weights is None:
        weights = torch.ones(max_n, device=candidates.device) / max_n

    max_n = weights.size(0)

    batch_size = candidates.size(0)
    num_refs = references.size(1)

    candidate_mask = (candidates!= pad_token_id)
    reference_mask = (references!= pad_token_id)

    candidate_lengths = candidate_mask.sum(dim=1)
    reference_lengths = reference_mask.sum(dim=2)

    numerators = torch.zeros(max_n, device=candidates.device, dtype=torch.long)
    denominators = torch.zeros(max_n, device=candidates.device, dtype=torch.long)

    for n in range(1, max_n + 1):
        if torch.any(candidate_lengths < n):
            continue

        candidate_ngrams = candidates.unfold(1, n, 1)
        valid_candidate_mask = candidate_mask.unfold(1, n, 1).all(dim=2)
        candidate_ngrams_flat = candidate_ngrams[valid_candidate_mask]

        reference_ngrams_list =[]
        for i in range(num_refs):
            ref_i = references[:, i, :]
            ref_mask_i = reference_mask[:, i, :]
            if torch.any(ref_mask_i.sum(dim=1) < n):
                reference_ngrams_list.append(torch.tensor([], dtype=torch.long, device=candidates.device).reshape(0, n))
                continue
            ref_ngrams_i = ref_i.unfold(1, n, 1)
            valid_ref_mask_i = ref_mask_i.unfold(1, n, 1).all(dim=2)
            reference_ngrams_list.append(ref_ngrams_i[valid_ref_mask_i])

        if candidate_ngrams_flat.size(0) == 0:
            continue

        reference_ngrams_flat = torch.cat(reference_ngrams_list, dim=0)

        all_ngrams = torch.cat([candidate_ngrams_flat, reference_ngrams_flat], dim=0)

        if all_ngrams.size(0) == 0:
            denominators[n-1] = candidate_ngrams_flat.size(0)
            continue

        unique_ngrams, inverse_indices = torch.unique(all_ngrams, dim=0, return_inverse=True)

        num_unique = unique_ngrams.size(0)

        cand_inv_indices = inverse_indices[:candidate_ngrams_flat.size(0)]
        ref_inv_indices = inverse_indices[candidate_ngrams_flat.size(0):]

        cand_batch_indices = torch.arange(batch_size, device=candidates.device).unsqueeze(1).expand_as(valid_candidate_mask)
        cand_batch_indices_flat = cand_batch_indices[valid_candidate_mask]

        cand_offsets = cand_batch_indices_flat * num_unique
        cand_bincount_indices = cand_inv_indices + cand_offsets

        candidate_counts = torch.bincount(
            cand_bincount_indices,
            minlength=batch_size * num_unique
        ).reshape(batch_size, num_unique)

        ref_max_counts = torch.zeros_like(candidate_counts)

        start_idx = 0
        for i in range(num_refs):
            ref_i = references[:, i, :]
            ref_mask_i = reference_mask[:, i, :]
            if torch.any(ref_mask_i.sum(dim=1) < n):
                continue

            valid_ref_mask_i = ref_mask_i.unfold(1, n, 1).all(dim=2)
            num_ref_ngrams_i = valid_ref_mask_i.sum()

            if num_ref_ngrams_i == 0:
                continue

            ref_inv_indices_i = ref_inv_indices[start_idx : start_idx + num_ref_ngrams_i]
            start_idx += num_ref_ngrams_i

            ref_batch_indices = torch.arange(batch_size, device=candidates.device).unsqueeze(1).expand_as(valid_ref_mask_i)
            ref_batch_indices_flat = ref_batch_indices[valid_ref_mask_i]

            ref_offsets = ref_batch_indices_flat * num_unique
            ref_bincount_indices = ref_inv_indices_i + ref_offsets

            ref_counts_i = torch.bincount(
                ref_bincount_indices,
                minlength=batch_size * num_unique
            ).reshape(batch_size, num_unique)

            ref_max_counts = torch.maximum(ref_max_counts, ref_counts_i)

        clipped_counts = torch.minimum(candidate_counts, ref_max_counts)

        numerators[n-1] = clipped_counts.sum()
        denominators[n-1] = candidate_ngrams_flat.size(0)

    precisions = torch.zeros(max_n, device=candidates.device)
    if smoothing_method == 'floor':
        safe_numerators = torch.where(numerators == 0, smoothing_epsilon, numerators)
        safe_denominators = torch.where(denominators == 0, 1.0, denominators)
        precisions = safe_numerators / safe_denominators
    elif smoothing_method == 'add-k':
        precisions = (numerators + smoothing_k) / (denominators + smoothing_k)
    elif smoothing_method == 'exp':
        for i in range(max_n):
            if denominators[i] > 0:
                precisions[i] = numerators[i] / denominators[i]
            else:
                precisions[i] = torch.tensor(0.0, device=candidates.device)
        for i in range(max_n):
            if precisions[i] == 0:
                if i == 0:
                    precisions[i] = torch.tensor(1e-12, device=candidates.device)
                else:
                    precisions[i] = precisions[i-1] / 2.0
    else: # 'none'
        if torch.any(denominators == 0):
            return torch.tensor(0.0, device=candidates.device)
        precisions = numerators / denominators

    closest_ref_lengths_indices = torch.abs(reference_lengths - candidate_lengths.unsqueeze(1)).argmin(dim=1)
    closest_ref_lengths = reference_lengths.gather(1, closest_ref_lengths_indices.unsqueeze(1)).squeeze(1)

    total_candidate_len = candidate_lengths.sum()
    total_closest_ref_len = closest_ref_lengths.sum()

    if total_candidate_len == 0:
        return torch.tensor(0.0, device=candidates.device)

    brevity_penalty = torch.tensor(1.0, device=candidates.device)
    if total_candidate_len < total_closest_ref_len:
        brevity_penalty = torch.exp(1 - total_closest_ref_len / total_candidate_len)

    log_precisions = torch.log(precisions + (1e-12 if smoothing_method == 'none' else 0)).to(candidates.device)
    score = brevity_penalty * torch.exp(torch.sum(weights.to(candidates.device) * log_precisions))

    return score