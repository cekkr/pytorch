import builtins
from typing import Optional, Tuple

import torch
from torch import TensorBase

class Parameter(TensorBase):
    def __init__(
        self,
        data: TensorBase = ...,
        requires_grad: builtins.bool = ...,
    ): ...

def is_lazy(param: TensorBase): ...

class UninitializedParameter(TensorBase):
    def __init__(
        self,
        data: TensorBase = ...,
        requires_grad: builtins.bool = ...,
    ): ...
    def materialize(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ): ...

class UninitializedBuffer(TensorBase):
    def __init__(
        self,
        data: TensorBase = ...,
        requires_grad: builtins.bool = ...,
    ): ...
    def materialize(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ): ...
