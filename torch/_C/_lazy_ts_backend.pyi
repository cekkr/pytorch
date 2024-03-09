# defined in torch/csrc/lazy/python/init.cpp

from typing import Any, List, Tuple

from torch import TensorBase

def _init(): ...
def _get_tensors_ts_device_data_node(
    tensors: List[TensorBase],
) -> Tuple[List[int], List[Any]]: ...
def _run_cached_graph(hash_str: str, graph_inputs: List[Any]) -> List[TensorBase]: ...
