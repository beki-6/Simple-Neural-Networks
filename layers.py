from __future__ import annotations
from typing import List, Optional, Callable
from random import random
from util import dot
from neurons import Neuron

class Layer:
    def __init__(self, previous_layer: Optional[Layer], num_neurons: int, 
                 learning_rate: float, activation_fn: Callable[[float], float], 
                 derivative_activation_fn: Callable[[float], float]) -> None:
        self.previous_layer: Optional[Layer] = previous_layer
        self.neurons: List[Neuron] = []
        for i in range(num_neurons):
            if previous_layer is None:
                random_weights: List[float] = []
            else:
                random_weights = [random()] * len(previous_layer.neurons)
            
            neuron: Neuron = Neuron(random_weights, learning_rate, activation_fn, derivative_activation_fn)
            self.neurons.append(neuron)
            self.output_cache: List[float] = [0.0] * num_neurons

    def output(self, inputs: List[float]) -> List[float]:
        if self.previous_layer is None:
            self.output_cache = inputs 
        else:
            self.output_cache = [n.output() for n in self.neurons]
        return self.output_cache
    
    def calculate_deltas_output_layer(self, expected: List[float]) -> None:
        for n in range(len(self.neurons)):
            self.neurons[n].delta = self.neurons[n].derivative_activation_fn(self.neurons[n].output_cache) * (expected[n] - self.output_cache[n])
    
    def calculate_deltas_hidden_layer(self, next_layer: Layer) -> None:
        for index, neuron in enumerate(self.neurons):
            next_weights: List[float] = [n.weights[index] for n in next_layer.neurons]
            next_deltas: List[float] = [n.delta for n in next_layer.neurons]
            sum_weights_deltas: float = dot(next_weights, next_deltas)
            neuron.delta = neuron.derivative_activation_fn(neuron.output_cache) * sum_weights_deltas