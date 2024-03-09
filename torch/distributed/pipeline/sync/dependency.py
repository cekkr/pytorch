# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Arbitrary dependency between two autograd lanes."""
from typing import List, Tuple

import torch
from torch import TensorBase

from .phony import get_phony

__all__: List[str] = ["fork", "Fork", "join", "Join"]


def fork(input: TensorBase) -> Tuple[TensorBase, TensorBase]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)

    return input, phony


class Fork(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Fork", input: TensorBase) -> Tuple[TensorBase, TensorBase]:  # type: ignore[override]
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: "Fork", grad_input: TensorBase, grad_grad: TensorBase) -> TensorBase:  # type: ignore[override]
        return grad_input


def join(input: TensorBase, phony: TensorBase) -> TensorBase:
    """Merge two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)

    return input


class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Join", input: TensorBase, phony: TensorBase) -> TensorBase:  # type: ignore[override]
        return input.detach()

    @staticmethod
    def backward(ctx: "Join", grad_input: TensorBase) -> Tuple[TensorBase, None]:  # type: ignore[override]
        return grad_input, None
