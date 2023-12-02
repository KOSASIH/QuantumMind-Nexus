class QuantumInspiredOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update_weights(self, weights, gradients):
        return weights - self.learning_rate * gradients
