# Import necessary libraries
from qiskit import QuantumCircuit, Aer, execute
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.aqua.algorithms import Grover

# Define the logical expression oracle
expression = 'Hello World'
oracle = LogicalExpressionOracle(expression)

# Create the Grover search circuit
grover_circuit = Grover(oracle)

# Set up the quantum simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(grover_circuit, backend)

# Execute the Grover search
result = job.result()
counts = result.get_counts(grover_circuit)
