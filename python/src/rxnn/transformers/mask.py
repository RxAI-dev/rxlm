import torch


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask for a given sequence length."""
    # Create a lower triangular matrix of ones
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    # Expand the mask to have the shape (1, 1, seq_len, seq_len)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask
