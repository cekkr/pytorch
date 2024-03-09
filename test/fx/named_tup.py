from typing import NamedTuple

import torch


class MyNamedTup(NamedTuple):
    i: torch.TensorBase
    f: torch.TensorBase
