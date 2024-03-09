import functools
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Sequence

import torch

import torch._decomp
import torch._prims

import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special
import torch.overrides

from torch._prims_common import torch_function_passthrough


@functools.lru_cache(None)
def torch_to_refs_map():
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
    modules = [
        (torch, torch._refs),
        (torch.nn, torch._refs.nn),
        (torch.nn.functional, torch._refs.nn.functional),
        (torch.special, torch._refs.special),
        (torch.fft, torch._refs.fft),
        (torch.linalg, torch._refs.linalg),
    ]
    r: Dict[Any, Any] = {
        torch.TensorBase.__invert__: torch._refs.bitwise_not,
        torch.TensorBase.__xor__: torch._refs.bitwise_xor,
        torch.TensorBase.__and__: torch._refs.bitwise_and,
        torch.TensorBase.__or__: torch._refs.bitwise_or,
        torch.TensorBase.__eq__: torch._refs.eq,
        torch.TensorBase.__rsub__: torch._refs.rsub,
        torch.TensorBase.__rtruediv__: torch._refs.rtruediv,
        torch.TensorBase.__floordiv__: torch._refs.floor_divide,
        torch.TensorBase.__rfloordiv__: torch._refs.rfloordiv,
        torch.TensorBase.__pow__: torch._refs.pow,
        torch.TensorBase.__rpow__: torch._refs.rpow,
        torch.TensorBase.new_empty: torch._refs.new_empty,
        torch.TensorBase.new_full: torch._refs.new_full,
        torch.TensorBase.new_zeros: torch._refs.new_zeros,
        torch.TensorBase.new_ones: torch._refs.new_ones,
        torch.TensorBase.fill_: torch._refs.fill_,
        torch.TensorBase.zero_: torch._refs.zero_,
        torch.TensorBase.to: torch._refs.to,
        torch.TensorBase.sum_to_size: torch._refs.sum_to_size,
        # TODO: Should these methods be mapped some other way?
        torch.TensorBase.copy_: torch._prims.copy_to,
        torch.TensorBase.resize: torch._prims.resize,
    }
    for mod_torch, mod_refs in modules:
        for s in mod_refs.__all__:  # type: ignore[attr-defined]
            r[mod_torch.__dict__.get(s)] = mod_refs.__dict__.get(s)

    # Support remapping torch.Tensor.foo to _refs.foo
    for s in dir(torch.TensorBase):
        if s in torch._refs.__all__:
            r[getattr(torch.TensorBase, s)] = torch._refs.__dict__.get(s)

    # Support conversions
    for s in torch._refs._conversions.__all__:
        tensor_attr = getattr(torch.TensorBase, s, None) or getattr(torch, s)
        r[tensor_attr] = torch._refs._conversions.__dict__.get(s)

    return r


@functools.lru_cache(None)
def all_prims():
    """
    Set of all prim functions, e.g., torch._prims.add in all_prims()
    """
    return {torch._prims.__dict__.get(s) for s in torch._prims.__all__}


class TorchRefsMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    use PrimTorch refs in torch._refs.  (Direct calls to _refs are unaffected.)

    >>> # xdoctest: +SKIP
    >>> with TorchRefsMode():
    ...     torch.add(x, y)  # calls torch._refs.add(x, y)

    By default, this context manager will fall back on the torch.* if the
    ref does not exist; set strict=True to error if this occurs.
    If the ref exists we still would like to fall back on the torch.* sometimes,
    this behavior can be customized by passing a function to should_fallback_fn.
    """

    def __init__(
        self,
        strict=False,
        should_fallback_fn=lambda *_: False,
        prims_mode_cls=nullcontext,
    ):
        self.strict = strict
        self.should_fallback_fn = should_fallback_fn
        self.prims_mode_cls = prims_mode_cls

    def __torch_function__(
        self,
        orig_func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
    ):
        if kwargs is None:
            kwargs = {}
        # For primitive operations, run them as is without interception
        # Unless we are in prims_mode, in which case we want to use nvprims
        if orig_func in torch_function_passthrough or orig_func in all_prims():
            with self.prims_mode_cls():
                return orig_func(*args, **kwargs)
        mapping = torch_to_refs_map()
        func = mapping.get(orig_func, None)

        # For torch.ops.aten.*, use registered decompositions from torch._decomp
        # torch._decomp.decomposition_table provides a mapping from
        # torch.ops.aten.* to torch._refs or torch._decomp.decompositions
        # implementations.
        # There're other ways to implement this functionality,
        # see https://github.com/pytorch/pytorch/pull/82657#discussion_r939776417
        if func is None and isinstance(orig_func, torch._ops.OpOverload):
            func = torch._decomp.decomposition_table.get(orig_func, None)

        if func is not None:
            # If the ref exists query whether we should use it or not
            if self.should_fallback_fn(self, orig_func, func, args, kwargs):
                return orig_func(*args, **kwargs)
            # torch calls inside func should be interpreted as refs calls
            with self:
                return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(
                f"no _refs support for {torch.overrides.resolve_name(orig_func)}"
            )
        return orig_func(*args, **kwargs)
