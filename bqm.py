# Import necessary libraries
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

# Define the optimization problem
bqm = BinaryQuadraticModel.from_ising({0: -1, 1: 1}, {(0, 1): 2})

# Set up the quantum annealer
sampler = EmbeddingComposite(DWaveSampler())

# Solve the optimization problem
response = sampler.sample(bqm, num_reads=10)

# Extract the optimal solution
optimal_solution = response.first.sample
