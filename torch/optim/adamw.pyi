from typing import Optional, Tuple, Union

from torch import TensorBase

from .optimizer import Optimizer, ParamsT

class AdamW(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, TensorBase] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ) -> None: ...
