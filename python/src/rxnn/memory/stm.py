import torch
import torch.nn as nn

class ShortTermMemory(nn.Module):
    """Short-term memory module for the Attention-based Memory System"""

    def __init__(self, num_layers: int, embed_dim: int, stm_size: int, init_type: str = 'normal',
                 is_trainable: bool = False, *args, **kwargs):
        super(ShortTermMemory, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.stm_size = stm_size
        self.batch_size = 1 # setting 1 as initial batch size (it will be normally used in inference/pre-training. Bigger batches are for RL stages)
        self.is_trainable = is_trainable
        assert init_type in ['normal', 'standard', 'uniform', 'ones', 'zeros'], \
            'STM init type must be one of "normal", "standard", "uniform", "ones", "zeros"'
        self.init_type = init_type
        stm = self._init_tensor()
        if self.is_trainable:
            self.memory = nn.Parameter(stm)
        else:
            self.register_buffer('memory', stm)

    def _init_tensor(self, init_type: str = None):
        init_type = init_type or self.init_type
        stm_shape = (self.num_layers, self.batch_size, self.stm_size, self.embed_dim)
        if init_type == 'normal':
            return torch.normal(0, 0.02, stm_shape)
        elif init_type == 'standard':
            return torch.normal(0, 1, stm_shape)
        elif init_type == 'uniform':
            return torch.rand(*stm_shape) * 0.02
        elif init_type == 'ones':
            return torch.ones(*stm_shape)
        else:
            return torch.zeros(*stm_shape)

    def forward(self, layer: int) -> torch.Tensor:
        return self.memory[layer]

    def update_layer(self, layer: int, new_stm: torch.Tensor):
        self.memory = self.memory.clone()
        self.memory[layer] = new_stm

    def update_all(self, new_stm: torch.Tensor):
        self.memory = new_stm

    def make_trainable(self):
        if not self.is_trainable:
            self.is_trainable = True
            initial_stm = self.memory.clone()
            delattr(self, 'memory')
            self.memory = nn.Parameter(initial_stm)

    def freeze(self):
        if self.is_trainable:
            self.requires_grad_(False)
            trained_stm = self.memory.clone()
            delattr(self, 'memory')
            self.register_buffer('memory', trained_stm)

    def reset(self, init_type: str = None):
        self.memory = self._init_tensor(init_type).to(self.memory.device)

    def clone_detach_reset(self):
        self.memory = self.memory.detach().clone()

    def resize(self, new_stm_size: int, init_type: str = None):
        self.stm_size = new_stm_size
        delattr(self, 'memory')
        self.register_buffer('memory', self._init_tensor(init_type))

    def batched_memory(self, batch_size: int, init_type: str = None):
        if init_type is not None:
            assert init_type in ['normal', 'standard', 'uniform', 'ones', 'zeros'], \
                'STM init type must be one of "normal", "standard", "uniform", "ones", "zeros"'
            self.init_type = init_type
        device = self.memory.device
        self.batch_size = batch_size
        delattr(self, 'memory')
        self.register_buffer('memory', self._init_tensor().to(device))

    def single_memory(self, init_type: str = None, use_mean_from_batch: bool = False):
        if init_type is not None:
            assert init_type in ['normal', 'standard', 'uniform', 'ones', 'zeros'], \
                'STM init type must be one of "normal", "standard", "uniform", "ones", "zeros"'
            self.init_type = init_type
        device = self.memory.device
        self.batch_size = 1
        if use_mean_from_batch:
            batch_mean = self.memory.mean(dim=(1, 2, 3), keepdim=True)
            delattr(self, 'memory')
            self.register_buffer('memory', batch_mean)
        else:
            delattr(self, 'memory')
            self.register_buffer('memory', self._init_tensor().to(device))
