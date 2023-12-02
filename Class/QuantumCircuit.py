import qiskit

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qc = qiskit.QuantumCircuit(num_qubits)
    
    def add_noise_model(self, noise_model):
        self.qc.noise_model = noise_model
    
    def apply_gate(self, gate, target_qubits):
        self.qc.append(gate, target_qubits)
    
    def measure(self, target_qubits):
        self.qc.measure(target_qubits, target_qubits)
    
    def simulate(self, shots=1024):
        backend = qiskit.Aer.get_backend('qasm_simulator')
        job = qiskit.execute(self.qc, backend, shots=shots)
        result = job.result().get_counts()
        return result
