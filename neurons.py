from util import dot
from typing import Callable, List

class Neuron:
    def __init__(self, weights: List[float], learning_rate: float, activation_fn: Callable[[float], float], derivative_activation_fn: Callable[[float], float]) -> None:
        self.weights: List[float] = weights
        self.activation_fn : Callable[[float], float] = activation_fn
        self.learning_rate: float = learning_rate
        self.derivative_activation_fn: Callable[[float], float] = derivative_activation_fn
        self.output_cache: float = 0.0
        self.delta: float = 0.0

    def output(self, inputs: List[float]) -> float:
        self.output_cache = dot(self.weights, inputs)
        return self.activation_fn(self.output_cache)