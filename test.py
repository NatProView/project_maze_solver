import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MazeEnv

# Script for testing distribution of reward values for DQN
grid = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
]
env = MazeEnv(grid)

def evaluate_sequence(env, sequence):
    env.reset()
    total_reward = 0
    for action in sequence:
        _, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

sequence_length = 20
num_sequences = 100
random_sequences = torch.randint(0, 4, (num_sequences, sequence_length), dtype=torch.int)
rewards = [evaluate_sequence(env, seq) for seq in random_sequences]

print(f"Mean Reward: {np.mean(rewards):.3f}, Std Dev: {np.std(rewards):.3f}")
plt.hist(rewards, bins=20, edgecolor="k")
plt.title("Distribution of Rewards for Random Sequences")
plt.xlabel("Cumulative Reward")
plt.ylabel("Frequency")
plt.show()

base_sequence = torch.randint(0, 4, (sequence_length,), dtype=torch.int)
perturbations = torch.randint(0, 4, (10, sequence_length), dtype=torch.int) 
gradient_rewards = [
    evaluate_sequence(env, base_sequence ^ perturbation) for perturbation in perturbations
]

print(f"Base Reward: {evaluate_sequence(env, base_sequence):.3f}")
print(f"Perturbed Rewards: {gradient_rewards}")
plt.plot(gradient_rewards, marker="o")
plt.title("Reward Variation with Perturbations")
plt.xlabel("Perturbation Index")
plt.ylabel("Reward")
plt.show()

def boundary_vs_interior(grid):
    boundary_seq = [2] * (len(grid[0]) - 1) + [3] * (len(grid) - 1)
    interior_seq = [2, 3] * (len(grid[0]) // 2)
    
    boundary_reward = evaluate_sequence(env, boundary_seq)
    interior_reward = evaluate_sequence(env, interior_seq)
    
    return boundary_reward, interior_reward

boundary_reward, interior_reward = boundary_vs_interior(grid)
print(f"Boundary Reward: {boundary_reward}, Interior Reward: {interior_reward}")
