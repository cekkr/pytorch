import torch

NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000


class SubTensor(torch.TensorBase):
    pass


class WithTorchFunction:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, torch.TensorBase):
            self._tensor = data
            return

        self._tensor = torch.tensor(data, requires_grad=requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        return WithTorchFunction(args[0]._tensor + args[1]._tensor)


class SubWithTorchFunction(torch.TensorBase):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        return super().__torch_function__(func, types, args, kwargs)
