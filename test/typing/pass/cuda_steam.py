import torch


def foo(x: torch.TensorBase):
    stream = torch.cuda.current_stream()
    x.record_stream(stream)
