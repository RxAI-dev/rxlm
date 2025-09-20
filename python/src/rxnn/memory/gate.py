import torch
import torch.nn as nn
from typing import TypeAlias, Literal

ResidualGateType: TypeAlias = Literal['static', 'elementwise', 'linear']
SlotStatusType: TypeAlias = Literal['mean', 'linear']

class ResidualGate(nn.Module):
    def __init__(
            self,
            stm_size: int,
            embed_dim: int,
            use_gate: bool = False,
            gate_type: ResidualGateType = 'static',
            per_slot_gate: bool = True,
            init_gate: float = 0.0,
            use_tanh_gate: bool = True,
            slot_status_type: SlotStatusType = 'mean',
            disable_residual: bool = False,
            **kwargs,
    ):
        super(ResidualGate, self).__init__(**kwargs)
        self.use_gate = use_gate
        self.per_slot_gate = per_slot_gate
        self.gate_type = gate_type
        self.use_tanh_gate = use_tanh_gate
        self.slot_status_type = slot_status_type
        self.disable_residual = disable_residual

        if self.use_gate:
            if self.gate_type == 'linear':
                self.gate = nn.Linear(stm_size, stm_size if self.per_slot_gate else 1)
            else:
                gate_shape = (stm_size, 1) if self.per_slot_gate else (1,)
                self.gate = nn.Parameter(torch.full(gate_shape, init_gate))
        else:
            self.gate = None

        self.gate_activation = nn.Tanh() if self.use_tanh_gate else nn.Sigmoid()

        if self.slot_status_type == 'linear':
            assert self.gate_type == 'linear' or self.per_slot_gate == True, \
                'Cannot use linear slot status with not per-slot elementwise gate'
            self.slot_status = nn.Linear(embed_dim, 1)
        else:
            self.slot_status = None

    def _slot_statuses(self, updated_stm: torch.Tensor):
        if self.gate_type == 'linear':
            if self.slot_status_type == 'linear':
                return self.slot_status(updated_stm).squeeze(-1)
            else:
                return updated_stm.mean(dim=-1)
        else:
            if self.per_slot_gate and self.slot_status_type == 'linear':
                return self.slot_status(updated_stm)
            else:
                mean_dim = -1 if self.per_slot_gate else [1, 2]
                return updated_stm.mean(dim=mean_dim, keepdim=True)

    def _dynamic_gate(self, old_value: torch.Tensor, new_value: torch.Tensor):
        if self.gate_type == 'linear':
            statuses = self._slot_statuses(new_value + old_value)
            gate_input = self.gate(statuses).unsqueeze(-1)
        else:
            gate_input = self.gate * self._slot_statuses(new_value + old_value)
        return self.gate_activation(gate_input)

    def _calculate_output(self, layer_gate: torch.Tensor, old_value: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
        if self.use_tanh_gate:
            return (1 + layer_gate) * new_value + (1 - layer_gate) * old_value
        else:
            return layer_gate * new_value + (1 - layer_gate) * old_value

    def forward(self, old_value: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
        if self.disable_residual:
            return new_value
        if not self.use_gate:
            return new_value + old_value

        if self.gate_type == 'static':
            layer_gate = self.gate_activation(self.gate)
        else:
            layer_gate = self._dynamic_gate(old_value, new_value)

        return self._calculate_output(layer_gate, old_value, new_value)
