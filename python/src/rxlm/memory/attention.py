import torch
import torch.nn as nn
from .stm import ShortTermMemory


class StmMemoryAttention(nn.Module):
    def __init__(
            self,
            stm: ShortTermMemory,
            attention_layers: nn.ModuleList,
            memory_norm_layers: nn.ModuleList,
            memory_input_norm_layers: nn.ModuleList,
            residual_gate_layers: nn.ModuleList,
            debug_mode: bool = False,
            debug_interval: int = 10,
            *args,
            **kwargs
    ):
        super(StmMemoryAttention, self).__init__(*args, **kwargs)
        self.stm = stm
        self.attention_layers = attention_layers
        self.memory_norm_layers = memory_norm_layers
        self.memory_input_norm_layers = memory_input_norm_layers
        self.residual_gate_layers = residual_gate_layers
        assert (len(self.attention_layers) == len(self.memory_norm_layers) ==
                len(self.residual_gate_layers) == len(self.memory_input_norm_layers) ==
                self.stm.memory.size(0))
        self.num_layers = len(attention_layers)
        self.debug_mode = debug_mode
        self.debug_interval = debug_interval
        self.debug_step = 0

    def update_max_len(self, max_seq_len: int):
        for i in range(self.num_layers):
            if self.attention_layers[i].rope is not None:
                self.attention_layers[i].rope.update_max_len(max_seq_len)

    def _main_attention(
            self,
            layer_idx: int,
            encoded_data: torch.Tensor,
            layer_stm: torch.Tensor,
            mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Normalize encoded layer data
        encoded_layer_data = self.memory_input_norm_layers[layer_idx](encoded_data[layer_idx])
        # 2. Normalize STM layer
        normalized_layer_stm = self.memory_norm_layers[layer_idx](layer_stm)

        # 3. Print normalization stats in debug mode
        if self.debug_mode and self.training:
            if self.debug_step != 0 and self.debug_step % self.debug_interval == 0:
                self.debug_step = 0
                print(
                    f"Normalized STM stats - mean: {normalized_layer_stm.mean().item():.4f}, std: {normalized_layer_stm.std().item():.4f}")
            else:
                self.debug_step += 1

        # 4. Calculate memory attention
        new_layer_stm = self.attention_layers[layer_idx](
            normalized_layer_stm,
            encoded_layer_data,
            encoded_layer_data,
            mask=mask
        )
        # 5. Combine new updated layer state with current STM state in residual gate
        return self.residual_gate_layers[layer_idx](layer_stm, new_layer_stm)  # residual

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Process correct attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        # 2. Init new empty STM
        new_stm = torch.zeros_like(self.stm.memory)

        # 3. Run Short-Term Memory update for all layers
        for i in range(self.num_layers):
            # 4. Get current layer STM value
            layer_stm = self.stm(i)

            # 5. Calculate memory attention and update layer
            new_stm[i] = self._main_attention(i, x, layer_stm, mask=attention_mask)
        # 6. Update all layers
        self.stm.update_all(new_stm)
        return self.stm.memory


class InterlayerStmMemoryAttention(StmMemoryAttention):
    def __init__(
            self,
            stm: ShortTermMemory,
            attention_layers: nn.ModuleList,
            memory_norm_layers: nn.ModuleList,
            memory_input_norm_layers: nn.ModuleList,
            residual_gate_layers: nn.ModuleList,
            mean_attention_layers: nn.ModuleList,
            mean_memory_norm_layers: nn.ModuleList,
            mean_residual_gate_layers: nn.ModuleList,
            mean_stm_norm: nn.Module,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs
    ):
        super(InterlayerStmMemoryAttention, self).__init__(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gate_layers,
            debug_mode=debug_mode, debug_interval=debug_interval, **kwargs
        )
        self.mean_attention_layers = mean_attention_layers
        self.mean_memory_norm_layers = mean_memory_norm_layers
        self.mean_stm_norm = mean_stm_norm
        self.mean_residual_gate_layers = mean_residual_gate_layers
        assert (len(self.mean_attention_layers) == len(self.mean_memory_norm_layers) ==
                len(self.mean_residual_gate_layers) == self.num_layers)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Process correct attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        # 2. Init new empty STM
        new_stm = torch.zeros_like(self.stm.memory)

        # 3. Get mean STM value from layers for mean interlayer memory attention
        mean_stm = self.stm.memory.mean(dim=0) # [batch_size, stm_size, embed_dim]
        # 4. Normalize mean STM layer
        normalized_mean_stm = self.mean_stm_norm(mean_stm)

        # 5. Run Short-Term Memory update for all layers
        for i in range(self.num_layers):
            # 6. Get current layer STM value
            layer_stm = self.stm(i)

            # 7. Mean interlayer memory attention
            # a) normalize STM layer value
            pre_normalized_layer_stm = self.mean_memory_norm_layers[i](layer_stm)
            # b) calculate attention between STM layer and mean value of all STM layers (from previous interaction)
            interlayer_stm = self.mean_attention_layers[i](pre_normalized_layer_stm, normalized_mean_stm, normalized_mean_stm, mask=None)
            # c) combine updated interlayer state with current STM state in residual gate
            updated_layer_stm = self.mean_residual_gate_layers[i](layer_stm, interlayer_stm)

            # 8. Main memory attention
            new_stm[i] = self._main_attention(i, x, updated_layer_stm, mask=attention_mask)
        # 9. Update all layers
        self.stm.update_all(new_stm)
        return self.stm.memory


class SelfStmMemoryAttention(StmMemoryAttention):
    def __init__(
            self,
            stm: ShortTermMemory,
            attention_layers: nn.ModuleList,
            memory_norm_layers: nn.ModuleList,
            memory_input_norm_layers: nn.ModuleList,
            residual_gate_layers: nn.ModuleList,
            self_attention_layers: nn.ModuleList,
            self_memory_norm_layers: nn.ModuleList,
            self_residual_gate_layers: nn.ModuleList,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs
    ):
        super(SelfStmMemoryAttention, self).__init__(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gate_layers,
            debug_mode=debug_mode, debug_interval=debug_interval, **kwargs
        )
        self.self_attention_layers = self_attention_layers
        self.self_memory_norm_layers = self_memory_norm_layers
        self.self_residual_gate_layers = self_residual_gate_layers
        assert (len(self.self_attention_layers) == len(self.self_memory_norm_layers) ==
                len(self.self_residual_gate_layers) == self.num_layers)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Process correct attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        # 2. Init new empty STM
        new_stm = torch.zeros_like(self.stm.memory)

        # 3. Run Short-Term Memory update for all layers
        for i in range(self.num_layers):
            # 4. Get current layer STM value
            layer_stm = self.stm(i)

            # 5. Memory Self-Attention
            # a) normalize STM layer value
            pre_normalized_layer_stm = self.self_memory_norm_layers[i](layer_stm)
            # b) calculate attention between STM layer and mean value of all STM layers (from previous interaction)
            self_layer_stm = self.self_attention_layers[i](
                pre_normalized_layer_stm,
                pre_normalized_layer_stm,
                pre_normalized_layer_stm,
                mask=None,
            )
            # c) combine updated interlayer state with current STM state in residual gate
            updated_layer_stm = self.self_residual_gate_layers[i](layer_stm, self_layer_stm)

            # 6. Main memory attention
            new_stm[i] = self._main_attention(i, x, updated_layer_stm, mask=attention_mask)
        # 7. Update all layers/models
        self.stm.update_all(new_stm)
        return self.stm.memory

class SelfInterlayerStmMemoryAttention(StmMemoryAttention):
    def __init__(
            self,
            stm: ShortTermMemory,
            attention_layers: nn.ModuleList,
            memory_norm_layers: nn.ModuleList,
            memory_input_norm_layers: nn.ModuleList,
            residual_gate_layers: nn.ModuleList,
            mean_attention_layers: nn.ModuleList,
            mean_memory_norm_layers: nn.ModuleList,
            mean_residual_gate_layers: nn.ModuleList,
            interlayer_gate_layers: nn.ModuleList,
            mean_stm_norm: nn.Module,
            debug_mode: bool = False,
            debug_interval: int = 10,
            **kwargs
    ):
        super(SelfInterlayerStmMemoryAttention, self).__init__(
            stm, attention_layers, memory_norm_layers, memory_input_norm_layers, residual_gate_layers,
            debug_mode=debug_mode, debug_interval=debug_interval, **kwargs
        )
        self.mean_attention_layers = mean_attention_layers
        self.mean_memory_norm_layers = mean_memory_norm_layers
        self.mean_stm_norm = mean_stm_norm
        self.mean_residual_gate_layers = mean_residual_gate_layers
        self.interlayer_gate_layers = interlayer_gate_layers
        assert (len(self.mean_attention_layers) == len(self.mean_memory_norm_layers) ==
                len(self.mean_residual_gate_layers) == len(self.interlayer_gate_layers) == self.num_layers)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Process correct attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
        # 2. Init new empty STM
        new_stm = torch.zeros_like(self.stm.memory)

        # 3. Get mean STM value from layers for mean interlayer memory attention
        mean_stm = self.stm.memory.mean(dim=0) # [batch_size, stm_size, embed_dim]
        # 4. Normalize mean STM layer
        normalized_mean_stm = self.mean_stm_norm(mean_stm)

        # 5. Run Short-Term Memory update for all layers
        for i in range(self.num_layers):
            # 6. Get current layer STM value
            layer_stm = self.stm(i)

            # 7. Gated self/interlayer memory attention
            # a) normalize STM layer value
            pre_normalized_layer_stm = self.mean_memory_norm_layers[i](layer_stm)
            # b) combine interlayer and current layer data with gate
            self_interlayer_stm_input = self.interlayer_gate_layers[i](pre_normalized_layer_stm, normalized_mean_stm)
            # c) calculate attention between STM layer and combined self/interlayer state
            self_interlayer_stm = self.mean_attention_layers[i](pre_normalized_layer_stm, self_interlayer_stm_input, self_interlayer_stm_input, mask=None)
            # d) combine updated interlayer state with current STM state in residual gate
            updated_layer_stm = self.mean_residual_gate_layers[i](layer_stm, self_interlayer_stm)

            # 8. Main memory attention
            new_stm[i] = self._main_attention(i, x, updated_layer_stm, mask=attention_mask)
        # 9. Update all layers/models
        self.stm.update_all(new_stm)
        return self.stm.memory

