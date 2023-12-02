import numpy as np

class QuantumInspiredLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)
    
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias
