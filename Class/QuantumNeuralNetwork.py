import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, Aer, execute

class QuantumNeuralNetwork:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params = np.random.rand(num_layers, num_qubits, 3)

    def _apply_single_qubit_gate(self, circuit, qubit, gate, params):
        theta, phi, lambd = params
        if gate == 'rx':
            circuit.rx(theta, qubit)
        elif gate == 'ry':
            circuit.ry(theta, qubit)
        elif gate == 'rz':
            circuit.rz(theta, qubit)
        elif gate == 'u3':
            circuit.u3(theta, phi, lambd, qubit)

    def _apply_cnot_gate(self, circuit, control, target):
        circuit.cx(control, target)

    def _apply_layers(self, circuit, params):
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                self._apply_single_qubit_gate(circuit, qubit, 'u3', params[layer][qubit])
            for qubit in range(self.num_qubits - 1):
                self._apply_cnot_gate(circuit, qubit, qubit + 1)

    def _measure(self, circuit):
        circuit.measure_all()

    def _simulate(self, circuit):
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit, simulator, shots=1000)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                circuit = QuantumCircuit(self.num_qubits)
                self._apply_layers(circuit, self.params)
                self._measure(circuit)
                counts = self._simulate(circuit)

                # Compute loss
                loss = 0
                for output in counts:
                    if output == y:
                        loss += counts[output]
                loss /= 1000

                # Update parameters
                gradients = np.zeros_like(self.params)
                for layer in range(self.num_layers):
                    for qubit in range(self.num_qubits):
                        for gate in range(3):
                            params_plus = self.params.copy()
                            params_plus[layer][qubit][gate] += np.pi / 2
                            circuit_plus = QuantumCircuit(self.num_qubits)
                            self._apply_layers(circuit_plus, params_plus)
                            self._measure(circuit_plus)
                            counts_plus = self._simulate(circuit_plus)
                            loss_plus = 0
                            for output in counts_plus:
                                if output == y:
                                    loss_plus += counts_plus[output]
                            loss_plus /= 1000

                            params_minus = self.params.copy()
                            params_minus[layer][qubit][gate] -= np.pi / 2
                            circuit_minus = QuantumCircuit(self.num_qubits)
                            self._apply_layers(circuit_minus, params_minus)
                            self._measure(circuit_minus)
                            counts_minus = self._simulate(circuit_minus)
                            loss_minus = 0
                            for output in counts_minus:
                                if output == y:
                                    loss_minus += counts_minus[output]
                            loss_minus /= 1000

                            gradients[layer][qubit][gate] = (loss_plus - loss_minus) / 2

                self.params -= learning_rate * gradients

    def predict(self, x_test):
        predictions = []
        for x in x_test:
            circuit = QuantumCircuit(self.num_qubits)
            self._apply_layers(circuit, self.params)
            self._measure(circuit)
            counts = self._simulate(circuit)
            prediction = max(counts, key=counts.get)
            predictions.append(prediction)
        return predictions

# Example usage
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array(['00', '01', '01', '10'])

qnn = QuantumNeuralNetwork(num_qubits=2, num_layers=2)
qnn.train(x_train, y_train, epochs=100, learning_rate=0.01)

x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = qnn.predict(x_test)
print(predictions)
