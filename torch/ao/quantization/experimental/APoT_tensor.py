import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer


# class to store APoT quantized tensor
class TensorAPoT:
    quantizer: APoTQuantizer
    data: torch.TensorBase

    def __init__(self, quantizer: APoTQuantizer, apot_data: torch.TensorBase):
        self.quantizer = quantizer
        self.data = apot_data

    def int_repr(self):
        return self.data
