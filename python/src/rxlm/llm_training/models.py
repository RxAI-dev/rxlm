import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class MLMHead(nn.Module, PyTorchModelHubMixin, license="apache-2.0"):
    def __init__(self, embed_dim: int, vocab_size: int, *args, **kwargs):
        super(MLMHead, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


class MLMTrainingModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            mlm_head: MLMHead,
            *args,
            **kwargs
    ):
        super(MLMTrainingModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.mlm_head = mlm_head

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        h, _ = self.encoder(x, attention_mask=attention_mask)
        y = self.mlm_head(h)
        return y

