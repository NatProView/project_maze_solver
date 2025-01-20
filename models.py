import numpy as np
import torch
from torch import nn, optim
from abc import ABC, abstractmethod
from collections import deque
import random
import matplotlib.pyplot as plt
import logging
import os
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class MazeEnv:
    def __init__(self, grid):
        self.grid = grid
        self.n_rows, self.n_cols = len(grid), len(grid[0])
        self.start = (0, 0)
        self.goal = (self.n_rows - 1, self.n_cols - 1)
        self.visited = set()
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.visited = set()
        return self.agent_pos

    def step(self, action):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        next_pos = (self.agent_pos[0] + moves[action][0], self.agent_pos[1] + moves[action][1])

        if 0 <= next_pos[0] < self.n_rows and 0 <= next_pos[1] < self.n_cols and self.grid[next_pos[0]][next_pos[1]] == 0:
            self.agent_pos = next_pos

        while not self.is_junction(self.agent_pos) and self.agent_pos != self.goal:
            possible_moves = self.get_possible_moves(self.agent_pos)
            if len(possible_moves) == 1:
                move = possible_moves[0]
                self.agent_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
            else:
                break

        distance_to_goal = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])

        if self.agent_pos == self.goal:
            reward = 10 
        elif self.agent_pos not in self.visited:
            reward = 1 - 0.01 * distance_to_goal 
        else:
            reward = -0.1 - 0.01 * distance_to_goal

        self.visited.add(self.agent_pos)

        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done

    def get_possible_moves(self, pos):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        possible_moves = []
        for move in moves:
            next_pos = (pos[0] + move[0], pos[1] + move[1])
            if 0 <= next_pos[0] < self.n_rows and 0 <= next_pos[1] < self.n_cols and self.grid[next_pos[0]][next_pos[1]] == 0:
                possible_moves.append(move)
        return possible_moves

    def is_junction(self, pos):
        return len(self.get_possible_moves(pos)) > 1


    def render(self):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) == self.agent_pos:
                    print("A", end=" ")
                elif self.grid[r][c] == 1:
                    print("#", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()
        



class Solver(ABC):
    def __init__(self, name):
        self.log_dir = "logs"
        self.name = name
        os.makedirs(self.log_dir, exist_ok=True) 

    @abstractmethod
    def train(self, env):
        pass

    @abstractmethod
    def solve(self, env):
        pass
    
    def log_to_file(self, message, filename="training_log.txt"):
        file_path = os.path.join(self.log_dir, filename)
        with open(file_path, "a") as log_file:
            log_file.write(message + "\n")
        logging.info(message)

    def save_plot(self, x, y, xlabel, ylabel, title, filename):
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        file_path = os.path.join(self.log_dir, filename)
        plt.savefig(file_path)
        plt.close()
        # logging.info(f"Wykres zapisany: {file_path}")

class GeneticSolver(Solver):
    def __init__(self, population_size=100, generations=500, mutation_rate=0.1, crossover_rate=0.2, elite_fraction=0.05, checkpoint_interval=250, name="generic_genetic"):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.checkpoint_interval = checkpoint_interval
        self.convergence_history = []
        self.log_dir = "logs/genetic"
        self.name = name
        self.solution_found = False

    def generate_population(self, solution_length):
        return torch.randint(0, 4, (self.population_size, solution_length), dtype=torch.int)

    def evaluate_population(self, env, population):
        fitness_scores = torch.zeros(self.population_size, dtype=torch.float)
        for i, individual in enumerate(population):
            fitness_scores[i] = self.evaluate_individual(env, individual)
        return fitness_scores

    def evaluate_individual(self, env, individual):
        env.reset()
        total_reward = 0.0
        for action in individual:
            _, reward, done = env.step(action.item())
            total_reward += reward
            if done:
                break
        return total_reward

    def tournament_selection(self, ranked_population, fitness_scores, tournament_size=3):
        selected = []
        for _ in range(len(ranked_population)):
            tournament_indices = torch.randint(0, len(ranked_population), (tournament_size,))
            best_idx = tournament_indices[torch.argmax(fitness_scores[tournament_indices])]
            selected.append(ranked_population[best_idx])
        return torch.stack(selected)

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.clone()
        cut = random.randint(1, len(parent1) - 1)
        return torch.cat((parent1[:cut], parent2[cut:]))

    def mutate(self, individual, mutation_rate):
        mutation_mask = torch.rand(len(individual)) < mutation_rate
        random_mutations = torch.randint(0, 4, (len(individual),), dtype=individual.dtype)
        individual[mutation_mask] = random_mutations[mutation_mask]
        return individual

    def train(self, env):
        solution_length = env.n_rows * env.n_cols
        population = self.generate_population(solution_length)
        num_elites = max(1, int(self.elite_fraction * self.population_size))

        for generation in range(self.generations):
            fitness_scores = self.evaluate_population(env, population)
            ranked_indices = torch.argsort(fitness_scores, descending=True)
            ranked_population = population[ranked_indices]

            elites = ranked_population[:num_elites]

            mutation_rate = max(0.01, self.mutation_rate * (1 - generation / self.generations))

            parents = self.tournament_selection(ranked_population, fitness_scores)

            new_population = []
            for _ in range(self.population_size - num_elites):
                parent1 = parents[random.randint(0, len(parents) - 1)]
                parent2 = parents[random.randint(0, len(parents) - 1)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate)
                new_population.append(child)

            population = torch.vstack([elites] + new_population)

            best_fitness = fitness_scores[ranked_indices[0]].item()
            self.convergence_history.append(best_fitness)

            print(f"Generation {generation}: Best fitness = {best_fitness:.3f}, Mutation rate = {mutation_rate:.3f}")

            if best_fitness >= 15.0 and env.agent_pos == env.goal:
                self.solution_found = True

                break

            message = f"Generation {generation}: Best fitness = {best_fitness:.3f}, Mutation rate = {mutation_rate:.3f}"
            self.log_to_file(message)
            
            if generation % self.checkpoint_interval == 0 and generation > 0:
                print(f"Checkpoint at generation {generation}: Saving state.")
                self.save(f"checkpoint_gen{generation}.pt", ranked_population[0])

        if self.solution_found:
            print("Solution found!")
        else:
            print("No solution found.")
            
        self.save_plot(
            x=range(len(self.convergence_history)),
            y=self.convergence_history,
            xlabel="Generation",
            ylabel="Best Fitness",
            title="Convergence of Genetic Algorithm",
            filename=f"{self.name}.png",
        )
        self.visualize_convergence()
        
        return ranked_population[0]

    def solve(self, env):
        best_individual = self.train(env)
        path = []
        env.reset()
        for action in best_individual:
            state, _, done = env.step(action.item())
            path.append(state)
            if done:
                print("Goal reached!")
                break
        return path

    def solve_with_individual(self, env, individual):
        path = []
        env.reset()
        for action in individual:
            state, _, done = env.step(action.item())
            path.append(state)
            if done:
                print("Goal reached!")
                break
        return path

    def save(self, file_path, best_individual):
        torch.save({
            'best_individual': best_individual,
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
        }, file_path)

    @staticmethod
    def load(file_path):
        checkpoint = torch.load(file_path)
        solver = GeneticSolver(
            population_size=checkpoint['population_size'],
            generations=checkpoint['generations'],
            mutation_rate=checkpoint['mutation_rate']
        )
        best_individual = checkpoint['best_individual']
        return solver, best_individual

    def visualize_convergence(self):
        plt.plot(self.convergence_history)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence of Genetic Algorithm")
        filename=f"{self.name}_vis.png"
        # plt.show()
        file_path = os.path.join(self.log_dir, filename)
        plt.savefig(file_path)
        plt.close()

    

class DQNSolver(Solver):
    def __init__(self, state_size=2, action_size=4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001, name="generic_dqn"):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.dqn = self._build_model()
        self.target_dqn = self._build_model()
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.replay_buffer = ReplayBuffer(100000)
        self.rewards_history = []
        self.name = name
        self.log_dir = "logs/dqn"

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def save(self, file_path):
        torch.save({
            'model_state': self.dqn.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'learning_rate': self.learning_rate,
        }, file_path)

    @staticmethod
    def load(file_path):
        checkpoint = torch.load(file_path)
        solver = DQNSolver(
            state_size=checkpoint['state_size'],
            action_size=checkpoint['action_size'],
            gamma=checkpoint['gamma'],
            epsilon=checkpoint['epsilon'],
            epsilon_decay=checkpoint['epsilon_decay'],
            epsilon_min=checkpoint['epsilon_min'],
            learning_rate=checkpoint['learning_rate']
        )
        solver.dqn.load_state_dict(checkpoint['model_state'])
        return solver
    
    def train(self, env, num_episodes=500, batch_size=64):
        optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

        for episode in range(num_episodes):
            state = np.array(env.reset(), dtype=np.float32)
            total_reward = 0

            for _ in range(100):
                if random.random() < self.epsilon:
                    action = random.randint(0, self.action_size - 1)
                else:
                    with torch.no_grad():
                        action = torch.argmax(self.dqn(torch.tensor(state))).item()

                next_state, reward, done = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done and env.agent_pos == env.goal:
                    print("Solution found!")
                    break

                if len(self.replay_buffer) >= batch_size:
                    self._replay(optimizer, batch_size)

            if episode % 10 == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.rewards_history.append(total_reward)
            print(f"Episode {episode}, Total Reward: {total_reward:.3f}, Epsilon: {self.epsilon:.2f}")
        
        self.plot_rewards()

        return self.dqn
            
    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Rewards")
        plt.legend()
        file_path = os.path.join(self.log_dir, "{self.name}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"Rewards plot saved to {file_path}")

    def _replay(self, optimizer, batch_size):
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer.sample(batch_size))
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_dqn(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def solve(self, env, max_steps=100):
        state = np.array(env.reset(), dtype=np.float32)
        path = [env.start]

        for _ in range(max_steps):
            with torch.no_grad():
                action = torch.argmax(self.dqn(torch.tensor(state))).item()

            next_state, _, done = env.step(action)
            path.append(env.agent_pos)

            if done:
                print("Goal reached!")
                break

            state = np.array(next_state, dtype=np.float32)

        return path
    
    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Rewards")
        plt.legend()
        file_path = os.path.join(self.log_dir, "dqn_rewards.png")
        plt.savefig(file_path)
        plt.close()
        print(f"Rewards plot saved to {self.name}.png")
    
class PSOSolver(Solver):
    def __init__(self, swarm_size=50, max_iterations=500, inertia=0.5, cognitive_coeff=1.5, social_coeff=1.5, checkpoint_interval=50, name="generic_pso"):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.checkpoint_interval = checkpoint_interval
        self.convergence_history = [] 
        self.log_dir = "logs/pso"
        self.name = name

    def initialize_swarm(self, solution_length):
        positions = torch.randint(0, 4, (self.swarm_size, solution_length), dtype=torch.int)
        velocities = torch.randn_like(positions, dtype=torch.float) * 2.0 

        return positions, velocities

    def evaluate_particle(self, env, particle):
        env.reset()
        total_reward = 0
        for action in particle:
            _, reward, done = env.step(action.item())
            total_reward += reward
            if done:
                total_reward += 100
                break
        return total_reward

    def evaluate_swarm(self, env, swarm):
        fitness_scores = torch.zeros(self.swarm_size, dtype=torch.float)
        for i, particle in enumerate(swarm):
            fitness_scores[i] = self.evaluate_particle(env, particle)
        return fitness_scores

    def train(self, env):
        solution_length = env.n_rows * env.n_cols
        swarm, velocities = self.initialize_swarm(solution_length)
        personal_best_positions = swarm.clone()
        personal_best_scores = self.evaluate_swarm(env, swarm)

        global_best_index = torch.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].clone()
        global_best_score = personal_best_scores[global_best_index].item()

        stagnation_counter = torch.zeros(self.swarm_size, dtype=torch.int)

        for iteration in range(self.max_iterations):
            diversity = swarm.float().std().item()

            if diversity < 0.5:
                print(f"Swarm is stagnating at iteration {iteration}. Introducing perturbations.")
                perturbation = torch.randint(0, 4, swarm.shape, dtype=torch.int)
                swarm = torch.where(torch.rand(swarm.shape) < 0.1, perturbation, swarm)

            for i in range(self.swarm_size):
                r1 = torch.rand(solution_length)
                r2 = torch.rand(solution_length)
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive_coeff * r1 * (personal_best_positions[i] - swarm[i]).float()
                    + self.social_coeff * r2 * (global_best_position - swarm[i]).float()
                )
                swarm[i] = (swarm[i].float() + velocities[i]).long()
                swarm[i] = torch.clamp(swarm[i], 0, 3)

                fitness = self.evaluate_particle(env, swarm[i])
                if fitness > personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = swarm[i].clone()
                    stagnation_counter[i] = 0 
                else:
                    stagnation_counter[i] += 1

            reset_indices = stagnation_counter > 20
            if reset_indices.any():
                print(f"Resetting {reset_indices.sum()} particles at iteration {iteration}.")
                perturbation = torch.randint(0, 4, (reset_indices.sum(), solution_length), dtype=torch.int)
                swarm[reset_indices] = perturbation
                velocities[reset_indices] = torch.zeros_like(swarm[reset_indices], dtype=torch.float)
                stagnation_counter[reset_indices] = 0

            global_best_index = torch.argmax(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index].clone()
            global_best_score = personal_best_scores[global_best_index].item()

            self.convergence_history.append(global_best_score)

            print(f"Iteration {iteration}: Best fitness = {global_best_score:.3f}, Diversity = {diversity:.3f}")

            if global_best_score >= 15.0: 
                solution_path = self.build_path(env, global_best_position)
                if solution_path[-1] == env.goal:
                    print("Solution found and validated: Goal reached!")
                    break
                else:
                    print("False positive: High fitness but did not reach goal.")

        self.visualize_convergence()
        return global_best_position

    def solve(self, env):
        best_solution = self.train(env)
        path = self.build_path(env, best_solution)
        print("Path found by PSO:", path)
        return path

    def solve_with_individual(self, env, individual):
        path = []
        env.reset()
        for action in individual:
            state, _, done = env.step(action.item())
            path.append(state)
            if done:
                print("Goal reached!")
                break
        return path

    def build_path(self, env, solution):
        path = []
        env.reset()
        for action in solution:
            state, _, done = env.step(action.item())
            path.append(state)
            if done:
                break
        return path

    def save(self, file_path, best_individual):
        torch.save({
            'best_individual': best_individual,
            'swarm_size': self.swarm_size,
            'max_iterations': self.max_iterations,
            'inertia': self.inertia,
            'cognitive_coeff': self.cognitive_coeff,
            'social_coeff': self.social_coeff,
        }, file_path)

    @staticmethod
    def load(file_path):
        checkpoint = torch.load(file_path)
        return PSOSolver(
            swarm_size=checkpoint['swarm_size'],
            max_iterations=checkpoint['max_iterations'],
            inertia=checkpoint['inertia'],
            cognitive_coeff=checkpoint['cognitive_coeff'],
            social_coeff=checkpoint['social_coeff'],
        )
        
        
    def visualize_convergence(self):
        plt.plot(self.convergence_history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title("Convergence of PSO Algorithm")
        filename=f"{self.name}_vis.png"
        # plt.show()
        file_path = os.path.join(self.log_dir, filename)
        plt.savefig(file_path)
        plt.close()
        

        
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    
SOLVERS = {}

def register_solver(name, solver_cls):
    SOLVERS[name] = solver_cls