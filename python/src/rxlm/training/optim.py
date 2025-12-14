"""
Hybrid Optimizers for RxT - Fixed version
Properly handles schedulers and gradient scaler
"""

import torch
from torch.optim import Optimizer


class MuonAdamW(Optimizer):
    """Wrapper: Muon + AdamW with proper scheduler/scaler support"""

    def __init__(self, params, muon_config=None, adamw_config=None):
        from muon import Muon

        # Separate parameters
        matrix_params = [p for p in params if p.requires_grad and p.ndim >= 2]
        non_matrix_params = [p for p in params if p.requires_grad and p.ndim < 2]

        muon_cfg = muon_config or {}
        adamw_cfg = adamw_config or {}

        # Create optimizers
        self.muon = Muon(
            matrix_params,
            lr=muon_cfg.get('lr', 0.02),
            momentum=muon_cfg.get('momentum', 0.95),
            nesterov=muon_cfg.get('nesterov', True),
            weight_decay=muon_cfg.get('weight_decay', 0.01),
        ) if matrix_params else None

        self.adamw = torch.optim.AdamW(
            non_matrix_params,
            lr=adamw_cfg.get('lr', 3e-4),
            betas=adamw_cfg.get('betas', (0.9, 0.95)),
            weight_decay=adamw_cfg.get('weight_decay', 0.01),
        ) if non_matrix_params else None

        # Don't call super().__init__ - we'll handle param_groups manually
        self.defaults = {}
        self._step_count = 0

    @property
    def param_groups(self):
        """Dynamic property that returns combined param groups"""
        groups = []
        if self.muon:
            groups.extend(self.muon.param_groups)
        if self.adamw:
            groups.extend(self.adamw.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        """Scheduler tries to set param_groups, we need to split and distribute"""
        muon_count = len(self.muon.param_groups) if self.muon else 0
        if self.muon and muon_count > 0:
            self.muon.param_groups = value[:muon_count]
        if self.adamw and len(value) > muon_count:
            self.adamw.param_groups = value[muon_count:]

    @property
    def state(self):
        """Combine states from both optimizers"""
        combined = {}
        if self.muon:
            combined.update(self.muon.state)
        if self.adamw:
            combined.update(self.adamw.state)
        return combined

    def step(self, closure=None):
        """Execute both optimizers"""
        loss = None
        if self.muon:
            loss = self.muon.step(closure)
        if self.adamw:
            self.adamw.step()
        self._step_count += 1
        return loss

    def zero_grad(self, set_to_none=True):
        if self.muon:
            self.muon.zero_grad(set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {
            'muon': self.muon.state_dict() if self.muon else None,
            'adamw': self.adamw.state_dict() if self.adamw else None,
            'step_count': self._step_count
        }

    def load_state_dict(self, state_dict):
        if self.muon and state_dict.get('muon'):
            self.muon.load_state_dict(state_dict['muon'])
        if self.adamw and state_dict.get('adamw'):
            self.adamw.load_state_dict(state_dict['adamw'])
        self._step_count = state_dict.get('step_count', 0)


class DionAdamW(Optimizer):
    """Wrapper: Dion + AdamW with proper scheduler/scaler support"""

    def __init__(self, params, dion_config=None, adamw_config=None):
        from dion import Dion

        matrix_params = [p for p in params if p.requires_grad and p.ndim >= 2]
        non_matrix_params = [p for p in params if p.requires_grad and p.ndim < 2]

        dion_cfg = dion_config or {}
        adamw_cfg = adamw_config or {}

        self.dion = Dion(
            matrix_params,
            lr=dion_cfg.get('lr', 0.02),
            weight_decay=dion_cfg.get('weight_decay', 0.01),
            momentum=dion_cfg.get('momentum', 0.95),
            rank_fraction=dion_cfg.get('rank_fraction', 0.5),
        ) if matrix_params else None

        self.adamw = torch.optim.AdamW(
            non_matrix_params,
            lr=adamw_cfg.get('lr', 3e-4),
            betas=adamw_cfg.get('betas', (0.9, 0.95)),
            weight_decay=adamw_cfg.get('weight_decay', 0.01),
        ) if non_matrix_params else None

        self.defaults = {}
        self._step_count = 0

    @property
    def param_groups(self):
        groups = []
        if self.dion:
            groups.extend(self.dion.param_groups)
        if self.adamw:
            groups.extend(self.adamw.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        dion_count = len(self.dion.param_groups) if self.dion else 0
        if self.dion and dion_count > 0:
            self.dion.param_groups = value[:dion_count]
        if self.adamw and len(value) > dion_count:
            self.adamw.param_groups = value[dion_count:]

    @property
    def state(self):
        combined = {}
        if self.dion:
            combined.update(self.dion.state)
        if self.adamw:
            combined.update(self.adamw.state)
        return combined

    def step(self, closure=None):
        loss = None
        if self.dion:
            loss = self.dion.step(closure)
        if self.adamw:
            self.adamw.step()
        self._step_count += 1
        return loss

    def zero_grad(self, set_to_none=True):
        if self.dion:
            self.dion.zero_grad(set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {
            'dion': self.dion.state_dict() if self.dion else None,
            'adamw': self.adamw.state_dict() if self.adamw else None,
            'step_count': self._step_count
        }

    def load_state_dict(self, state_dict):
        if self.dion and state_dict.get('dion'):
            self.dion.load_state_dict(state_dict['dion'])
        if self.adamw and state_dict.get('adamw'):
            self.adamw.load_state_dict(state_dict['adamw'])
        self._step_count = state_dict.get('step_count', 0)


class NorMuonAdamW(Optimizer):
    """Wrapper: NorMuon + AdamW with proper scheduler/scaler support"""

    def __init__(self, params, normuon_config=None, adamw_config=None):
        from normuon import NorMuon

        matrix_params = [p for p in params if p.requires_grad and p.ndim >= 2]
        non_matrix_params = [p for p in params if p.requires_grad and p.ndim < 2]

        normuon_cfg = normuon_config or {}
        adamw_cfg = adamw_config or {}

        self.normuon = NorMuon(
            matrix_params,
            lr=normuon_cfg.get('lr', 0.02),
            weight_decay=normuon_cfg.get('weight_decay', 0.01),
            muon_momentum=normuon_cfg.get('muon_momentum', 0.95),
            adam_beta2=normuon_cfg.get('adam_beta2', 0.95),
        ) if matrix_params else None

        self.adamw = torch.optim.AdamW(
            non_matrix_params,
            lr=adamw_cfg.get('lr', 3e-4),
            betas=adamw_cfg.get('betas', (0.9, 0.95)),
            weight_decay=adamw_cfg.get('weight_decay', 0.01),
        ) if non_matrix_params else None

        self.defaults = {}
        self._step_count = 0

    @property
    def param_groups(self):
        groups = []
        if self.normuon:
            groups.extend(self.normuon.param_groups)
        if self.adamw:
            groups.extend(self.adamw.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        normuon_count = len(self.normuon.param_groups) if self.normuon else 0
        if self.normuon and normuon_count > 0:
            self.normuon.param_groups = value[:normuon_count]
        if self.adamw and len(value) > normuon_count:
            self.adamw.param_groups = value[normuon_count:]

    @property
    def state(self):
        combined = {}
        if self.normuon:
            combined.update(self.normuon.state)
        if self.adamw:
            combined.update(self.adamw.state)
        return combined

    def step(self, closure=None):
        loss = None
        if self.normuon:
            loss = self.normuon.step(closure)
        if self.adamw:
            self.adamw.step()
        self._step_count += 1
        return loss

    def zero_grad(self, set_to_none=True):
        if self.normuon:
            self.normuon.zero_grad(set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {
            'normuon': self.normuon.state_dict() if self.normuon else None,
            'adamw': self.adamw.state_dict() if self.adamw else None,
            'step_count': self._step_count
        }

    def load_state_dict(self, state_dict):
        if self.normuon and state_dict.get('normuon'):
            self.normuon.load_state_dict(state_dict['normuon'])
        if self.adamw and state_dict.get('adamw'):
            self.adamw.load_state_dict(state_dict['adamw'])
        self._step_count = state_dict.get('step_count', 0)