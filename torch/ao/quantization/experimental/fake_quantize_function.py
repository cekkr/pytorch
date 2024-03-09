import torch
from torch import TensorBase
from torch.ao.quantization.experimental.quantizer import dequantize_APoT, quantize_APoT


class fake_quantize_function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,  # type: ignore[override]
        x: TensorBase,
        alpha: TensorBase,
        gamma: TensorBase,
        quantization_levels: TensorBase,
        level_indices: TensorBase,
    ) -> TensorBase:
        quantized_result = quantize_APoT(
            x, alpha, gamma, quantization_levels, level_indices
        )

        # calculate mask tensor
        mask = x.detach().apply_(lambda x: (x <= alpha and x >= -alpha))

        result = dequantize_APoT(quantized_result)

        ctx.save_for_backward(mask)

        return result

    @staticmethod
    def backward(ctx, grad_output: TensorBase) -> TensorBase:  # type: ignore[override]
        mask = ctx.saved_tensors
        return grad_output * mask
