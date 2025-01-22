import os
import random
import logging
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv

from abc import ABC, abstractmethod
from collections import deque
from torch import nn, optim

matplotlib.use('Agg') 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# TODO moze pomoc is junction genetycznym i pso, tylko decyzje na skrzyzowaniach

# ==============================================================================
#                             MAZE ENVIRONMENTS
# ==============================================================================

class MazeEnvDQN:
    def __init__(self, grid):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
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
        next_pos = (self.agent_pos[0] + moves[action][0],
                    self.agent_pos[1] + moves[action][1])

        if (0 <= next_pos[0] < self.n_rows and
            0 <= next_pos[1] < self.n_cols and
            self.grid[next_pos[0]][next_pos[1]] == 0):
            self.agent_pos = next_pos

        while not self.is_junction(self.agent_pos) and self.agent_pos != self.goal:
            possible_moves = self.get_possible_moves(self.agent_pos)
            if len(possible_moves) == 1:
                move = possible_moves[0]
                self.agent_pos = (self.agent_pos[0] + move[0],
                                  self.agent_pos[1] + move[1])
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
        done = (self.agent_pos == self.goal)
        return self.agent_pos, reward, done

    def get_possible_moves(self, pos):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        possible_moves = []
        for move in moves:
            next_pos = (pos[0] + move[0], pos[1] + move[1])
            if (0 <= next_pos[0] < self.n_rows and
                0 <= next_pos[1] < self.n_cols and
                self.grid[next_pos[0]][next_pos[1]] == 0):
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


class MazeEnvGenetic:
    def __init__(self, grid):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.start = (0, 0)
        self.goal = (self.n_rows - 1, self.n_cols - 1)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.visited = set([self.agent_pos])
        self.prev_distance = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
        return self.agent_pos

    def get_possible_moves(self, pos):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        possible_moves = []
        for move in moves:
            nx = pos[0] + move[0]
            ny = pos[1] + move[1]
            if 0 <= nx < self.n_rows and 0 <= ny < self.n_cols and self.grid[nx][ny] == 0:
                possible_moves.append(move)
        return possible_moves

    def is_junction(self, pos):
        return len(self.get_possible_moves(pos)) > 1

    def step(self, action):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        dx, dy = moves[action]
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        distance_before = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
        if (0 <= next_pos[0] < self.n_rows and
            0 <= next_pos[1] < self.n_cols and
            self.grid[next_pos[0]][next_pos[1]] == 0):
            self.agent_pos = next_pos

        while not self.is_junction(self.agent_pos) and self.agent_pos != self.goal:
            possible_moves = self.get_possible_moves(self.agent_pos)
            if len(possible_moves) == 1:
                move = possible_moves[0]
                self.agent_pos = (self.agent_pos[0] + move[0],
                                  self.agent_pos[1] + move[1])
            else:
                break

        distance_after = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])

        done = False
        if self.agent_pos == self.goal:
            reward = 20.0
            done = True
        else:
            reward = 0.0
            if self.agent_pos in self.visited:
                reward -= 1.0
            if distance_after < distance_before:
                reward += 0.3
            else:
                reward -= 0.1

        self.visited.add(self.agent_pos)
        self.prev_distance = distance_after
        return self.agent_pos, reward, done

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



class MazeEnvPSO:
    def __init__(self, grid):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.start = (0, 0)
        self.goal = (self.n_rows - 1, self.n_cols - 1)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.visited = set([self.agent_pos])
        self.prev_distance = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
        return self.agent_pos

    def get_possible_moves(self, pos):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        possible_moves = []
        for move in moves:
            nx = pos[0] + move[0]
            ny = pos[1] + move[1]
            if 0 <= nx < self.n_rows and 0 <= ny < self.n_cols and self.grid[nx][ny] == 0:
                possible_moves.append(move)
        return possible_moves

    def is_junction(self, pos):
        return len(self.get_possible_moves(pos)) > 1

    def step(self, action):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        dx, dy = moves[action]
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        if (0 <= next_pos[0] < self.n_rows and
            0 <= next_pos[1] < self.n_cols and
            self.grid[next_pos[0]][next_pos[1]] == 0):
            self.agent_pos = next_pos

        while not self.is_junction(self.agent_pos) and self.agent_pos != self.goal:
            possible_moves = self.get_possible_moves(self.agent_pos)
            if len(possible_moves) == 1:
                move = possible_moves[0]
                self.agent_pos = (self.agent_pos[0] + move[0],
                                  self.agent_pos[1] + move[1])
            else:
                break

        done = False
        distance_to_goal = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])

        reward = 0.0
        if self.agent_pos == self.goal:
            reward += 50.0
            done = True
        else:
            if self.agent_pos in self.visited:
                reward -= 1.0

            if distance_to_goal < self.prev_distance:
                reward += 0.2
            elif distance_to_goal > self.prev_distance:
                reward -= 0.1

        self.visited.add(self.agent_pos)
        self.prev_distance = distance_to_goal
        return self.agent_pos, reward, done

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

# ==============================================================================
#                             SOLVER BASE CLASS
# ==============================================================================

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

    def calculate_metrics(self, values):
        mean_value = np.mean(values)
        std_dev = np.std(values)
        skewness = stats.skew(values)
        return {
            "mean": mean_value,
            "std_dev": std_dev,
            "skewness": skewness,
        }

    def log_to_file(self, message, filename):
        print(f"Opening {filename}")
        with open(filename, "a") as log_file:
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


# ==============================================================================
#                             REPLAY BUFFER
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
#                             DQN SOLVER
# ==============================================================================
class DQNSolver(Solver):
    def __init__(
        self,
        state_size=2,
        action_size=4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.1,
        learning_rate=0.001,
        replay_buffer_size=100000,
        name="generic_dqn"
    ):
        super().__init__(name=name)
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
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.rewards_history = []

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def train(self, env, num_episodes=500, batch_size=64, max_steps=100):
        optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

        log_file = os.path.join("logs_detailed/dqn", f"{self.name}_dqn_log.csv")
        os.makedirs("logs_detailed/dqn", exist_ok=True)
        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode", "total_reward", "epsilon"])
            
            for episode in range(num_episodes):
                state = np.array(env.reset(), dtype=np.float32)
                total_reward = 0

                for _ in range(max_steps):
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
                    writer.writerow([episode, total_reward, self.epsilon])
                    if done:
                        break

                    if len(self.replay_buffer) >= batch_size:
                        self._replay(optimizer, batch_size)

                if episode % 10 == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())

                
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                self.rewards_history.append(total_reward)

                print(f"[Episode {episode}] Reward: {total_reward:.2f}, Eps: {self.epsilon:.2f}")

        self.plot_rewards()
        metrics = self.calculate_metrics(self.rewards_history)
        
        metrics_file = os.path.join("metrics/dqn", f"{self.name}.txt")
        os.makedirs("metrics/dqn", exist_ok=True)
        self.log_to_file(f"Metrics: {metrics}", metrics_file)
        
        return self.dqn, metrics

    def _replay(self, optimizer, batch_size):
        transitions = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
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

            next_state, reward, done = env.step(action)
            path.append(env.agent_pos)

            if done:
                print("Goal reached by DQNSolver!")
                break

            state = np.array(next_state, dtype=np.float32)

        return path

    def plot_rewards(self):
        plt.figure()
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"DQN Training Rewards - {self.name}")
        plt.legend()
        file_path = os.path.join(self.log_dir, "dqn", f"{self.name}.png")
        os.makedirs(os.path.join(self.log_dir, "dqn"), exist_ok=True)
        plt.savefig(file_path)
        plt.close()

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
            'replay_buffer_size': self.replay_buffer  
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
            learning_rate=checkpoint['learning_rate'],
            replay_buffer_size=checkpoint.get('replay_buffer_size', 100000)
        )
        solver.dqn.load_state_dict(checkpoint['model_state'])
        solver.target_dqn.load_state_dict(solver.dqn.state_dict())
        return solver


class DoubleDQNSolver(Solver):
    def __init__(
        self,
        state_size=2,
        action_size=4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        learning_rate=0.0005,
        replay_buffer_size=100000, 
        tau=0.01,
        name="double_dqn"
    ):
        super().__init__(name=name)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.tau = tau
        self.name = name

        self.dqn = self._build_model()
        self.target_dqn = self._build_model()
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.replay_buffer = deque(maxlen=200_000)
        self.rewards_history = []

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def soft_update(self):
        for target_param, main_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(
                self.tau * main_param.data + (1 - self.tau) * target_param.data
            )

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experience(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def train(self, env, num_episodes=500, batch_size=64, max_steps=100):
        optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        log_file = os.path.join("logs_detailed/double_dqn", f"{self.name}_double_dqn_log.csv")
        os.makedirs("logs_detailed/double_dqn", exist_ok=True)
        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode", "total_reward", "epsilon"])
            
            for episode in range(num_episodes):
                state = np.array(env.reset(), dtype=np.float32)
                total_reward = 0

                for _ in range(max_steps):
                    if random.random() < self.epsilon:
                        action = random.randint(0, self.action_size - 1)
                    else:
                        with torch.no_grad():
                            action = torch.argmax(self.dqn(torch.tensor(state))).item()

                    next_state, reward, done = env.step(action)
                    next_state = np.array(next_state, dtype=np.float32)

                    self.push_experience(state, action, reward, next_state, done)

                    state = next_state
                    total_reward += reward
                    writer.writerow([episode, total_reward, self.epsilon])
                    if done:
                        break

                    if len(self.replay_buffer) >= batch_size:
                        self._replay(optimizer, batch_size)

                self.soft_update()

                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                self.rewards_history.append(total_reward)

                print(f"[Episode {episode}] Reward: {total_reward:.2f}, Eps: {self.epsilon:.2f}")
                
            self.plot_rewards()
            metrics = self.calculate_metrics(self.rewards_history)
            metrics_file = os.path.join("metrics/double_dqn", f"{self.name}.txt")
            os.makedirs("metrics/double_dqn", exist_ok=True)
            self.log_to_file(f"Metrics: {metrics}", metrics_file)
            return self.dqn, metrics

    def _replay(self, optimizer, batch_size):
        transitions = self.sample_experience(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.dqn(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_dqn(next_states).gather(1, next_actions).squeeze(1)

        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def solve(self, env, max_steps=300):
        state = np.array(env.reset(), dtype=np.float32)
        path = [env.start]

        for _ in range(max_steps):
            with torch.no_grad():
                action = torch.argmax(self.dqn(torch.tensor(state))).item()

            next_state, reward, done = env.step(action)
            path.append(env.agent_pos)

            if done:
                print("Goal reached by Double DQN!")
                break

            state = np.array(next_state, dtype=np.float32)

        return path
    
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
            'tau': self.tau,
            'replay_buffer_size': self.replay_buffer_size
        }, file_path)
        
    @staticmethod
    def load(file_path):
        checkpoint = torch.load(file_path)
        solver = DoubleDQNSolver(
            state_size=checkpoint['state_size'],
            action_size=checkpoint['action_size'],
            gamma=checkpoint['gamma'],
            epsilon=checkpoint['epsilon'],
            epsilon_decay=checkpoint['epsilon_decay'],
            epsilon_min=checkpoint['epsilon_min'],
            learning_rate=checkpoint['learning_rate'],
            tau=checkpoint.get('tau', 0.01),
            replay_buffer_size=checkpoint.get('replay_buffer_size', 100000)
        )
        solver.dqn.load_state_dict(checkpoint['model_state'])
        solver.target_dqn.load_state_dict(solver.dqn.state_dict())
        return solver
    
    def plot_rewards(self):
        plt.figure()
        plt.plot(self.rewards_history, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Double DQN Training Rewards - {self.name}")
        plt.legend()
        file_path = os.path.join(self.log_dir, "double_dqn", f"{self.name}.png")
        os.makedirs(os.path.join(self.log_dir, "double_dqn"), exist_ok=True)
        plt.savefig(file_path)
        plt.close()
        
        
# ==============================================================================
#                          GENETIC SOLVER
# ==============================================================================
class GeneticSolver(Solver):
    def __init__(
        self,
        population_size=200,
        generations=500,
        mutation_rate=0.1,
        crossover_rate=0.2,
        elite_fraction=0.02,
        checkpoint_interval=250,
        name="generic_genetic"
    ):
        super().__init__(name=name)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.checkpoint_interval = checkpoint_interval
        self.convergence_history = []
        self.solution_found = False
        self.best_individual = None

        self.stagnation_threshold = 30 
        self.immigrant_interval = 20 

    def generate_population(self, solution_length):
        return torch.randint(0, 4, (self.population_size, solution_length), dtype=torch.int)

    def evaluate_individual(self, env, individual):
        env.reset()
        total_reward = 0.0
        for action in individual:
            _, reward, done = env.step(action.item())
            total_reward += reward
            if done:
                break
        return total_reward

    def evaluate_population(self, env, population):
        fitness_scores = torch.zeros(self.population_size, dtype=torch.float)
        for i, individual in enumerate(population):
            fitness_scores[i] = self.evaluate_individual(env, individual)
        return fitness_scores

    def tournament_selection(self, ranked_population, fitness_scores, tournament_size=2):
        selected = []
        pop_len = len(ranked_population)
        for _ in range(pop_len):
            tournament_indices = torch.randint(0, pop_len, (tournament_size,))
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
        solution_length = 2 * (env.n_rows + env.n_cols)
        population = self.generate_population(solution_length)

        num_elites = max(1, int(self.elite_fraction * self.population_size))
        log_dir = os.path.join("logs_detailed", "genetic")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.name}_genetic_log.csv")

        last_best_fitness = float('-inf')
        last_improvement_gen = 0

        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["generation", "best_fitness"])

            for generation in range(self.generations):
                fitness_scores = self.evaluate_population(env, population)
                ranked_indices = torch.argsort(fitness_scores, descending=True)
                ranked_population = population[ranked_indices]

                best_fitness = fitness_scores[ranked_indices[0]].item()
                self.convergence_history.append(best_fitness)
                writer.writerow([generation, best_fitness])
                self.best_individual = ranked_population[0].clone()

                print(f"Generation {generation}: Best fitness = {best_fitness:.3f}")

                if best_fitness > last_best_fitness:
                    last_best_fitness = best_fitness
                    last_improvement_gen = generation

                if best_fitness >= 20.0:
                    self.solution_found = True
                    print("Solution found!")
                    break

                elites = ranked_population[:num_elites]

                mutation_rate = max(0.01, self.mutation_rate * (1 - generation / self.generations))

                parents = self.tournament_selection(ranked_population, fitness_scores, tournament_size=2)
                new_population = []
                for _ in range(self.population_size - num_elites):
                    parent1 = parents[random.randint(0, len(parents) - 1)]
                    parent2 = parents[random.randint(0, len(parents) - 1)]
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child, mutation_rate)
                    new_population.append(child)

                population = torch.vstack([elites] + new_population)

                if generation > 0 and generation % self.immigrant_interval == 0:
                    immigrant_count = 5
                    worst_indices = ranked_indices[-immigrant_count:]
                    for idx in worst_indices:
                        population[idx] = torch.randint(0, 4, (solution_length,), dtype=torch.int)
                    print(f"Random immigrants introduced at generation {generation}.")

                if (generation - last_improvement_gen) > self.stagnation_threshold:
                    print(f"Stagnation detected at generation {generation}, resetting 10% of population.")
                    reset_count = int(0.1 * self.population_size)
                    reset_indices = random.sample(range(num_elites, self.population_size), reset_count)
                    for idx in reset_indices:
                        population[idx] = torch.randint(0, 4, (solution_length,), dtype=torch.int)
                    last_improvement_gen = generation 

                if generation % self.checkpoint_interval == 0 and generation > 0:
                    ckpt_path = f"{self.name}_checkpoint_gen{generation}.pt"
                    self.save(ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")

        os.makedirs(os.path.join(self.log_dir, "genetic"), exist_ok=True)
        self.save_plot(
            x=range(len(self.convergence_history)),
            y=self.convergence_history,
            xlabel="Generation",
            ylabel="Best Fitness",
            title="Convergence of Genetic Algorithm",
            filename=f"genetic/{self.name}.png",
        )

        metrics = self.calculate_metrics(self.convergence_history)
        metrics_dir = os.path.join("metrics", "genetic")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f"{self.name}.txt")
        self.log_to_file(f"Metrics: {metrics}", filename=metrics_file)

        return self.best_individual, metrics

    def solve(self, env, max_steps=None):
        if self.best_individual is None:
            raise ValueError("Brak najlepszego osobnika! Najpierw wytrenuj solver lub wczytaj model.")

        path = []
        state = env.reset()
        for idx, action in enumerate(self.best_individual):
            if max_steps is not None and idx >= max_steps:
                break

            next_state, reward, done = env.step(action.item())
            path.append(next_state)
            if done:
                print("Goal reached by GeneticSolver!")
                break

        return path

    def solve_with_individual(self, env, individual, max_steps=None):
        path = []
        env.reset()
        for idx, action in enumerate(individual):
            if max_steps is not None and idx >= max_steps:
                break

            state, reward, done = env.step(action.item())
            path.append(state)
            if done:
                print("Goal reached by GeneticSolver!")
                break
        return path

    def save(self, file_path):
        torch.save({
            'best_individual': self.best_individual,
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elite_fraction': self.elite_fraction,
            'convergence_history': self.convergence_history
        }, file_path)

    @staticmethod
    def load(file_path):
        checkpoint = torch.load(file_path)
        solver = GeneticSolver(
            population_size=checkpoint['population_size'],
            generations=checkpoint['generations'],
            mutation_rate=checkpoint['mutation_rate'],
            crossover_rate=checkpoint.get('crossover_rate', 0.2),
            elite_fraction=checkpoint.get('elite_fraction', 0.05),
            name="loaded_genetic"
        )
        solver.best_individual = checkpoint['best_individual']
        solver.convergence_history = checkpoint.get('convergence_history', [])
        return solver, solver.best_individual


# ==============================================================================
#                             PSO SOLVER
# ==============================================================================
class PSOSolver(Solver):
    def __init__(
        self,
        swarm_size=50,
        max_iterations=500,
        inertia=0.5,
        cognitive_coeff=1.5,
        social_coeff=1.5,
        checkpoint_interval=50,
        name="generic_pso"
    ):
        super().__init__(name=name)
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations

        self.start_inertia = inertia
        self.end_inertia = 0.4
        
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.checkpoint_interval = checkpoint_interval
        
        self.convergence_history = []
        self.best_solution = None

    def initialize_swarm(self, solution_length):
        positions = torch.randint(0, 4, (self.swarm_size, solution_length), dtype=torch.int)
        velocities = torch.zeros_like(positions, dtype=torch.float)
        return positions, velocities

    def evaluate_particle(self, env, particle):
        env.reset()
        total_reward = 0
        for action in particle:
            _, reward, done = env.step(action.item())
            total_reward += reward
            if done:
                break
        return total_reward

    def evaluate_swarm(self, env, swarm):
        fitness_scores = torch.zeros(self.swarm_size, dtype=torch.float)
        for i, particle in enumerate(swarm):
            fitness_scores[i] = self.evaluate_particle(env, particle)
        return fitness_scores

    def train(self, env):
        solution_length = 2 * (env.n_rows + env.n_cols)
        swarm, velocities = self.initialize_swarm(solution_length)

        personal_best_positions = swarm.clone()
        personal_best_scores = self.evaluate_swarm(env, swarm)

        global_best_index = torch.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].clone()
        global_best_score = personal_best_scores[global_best_index].item()

        last_improvement = 0
        best_score_so_far = global_best_score

        log_dir = os.path.join("logs_detailed/pso")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.name}_pso_log.csv")

        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["iteration", "best_fitness"])

            for iteration in range(self.max_iterations):
                progress = iteration / self.max_iterations
                self.inertia = self.start_inertia - progress * (self.start_inertia - self.end_inertia)

                for i in range(self.swarm_size):
                    r1 = torch.rand(solution_length)
                    r2 = torch.rand(solution_length)

                    velocities[i] = (
                        self.inertia * velocities[i]
                        + self.cognitive_coeff * r1 * (personal_best_positions[i] - swarm[i]).float()
                        + self.social_coeff * r2 * (global_best_position - swarm[i]).float()
                    )

                    new_position = swarm[i].float() + velocities[i]
                    new_position = torch.clamp(new_position, 0, 3).long()

                    swarm[i] = new_position

                    fitness = self.evaluate_particle(env, swarm[i])
                    if fitness > personal_best_scores[i]:
                        personal_best_scores[i] = fitness
                        personal_best_positions[i] = swarm[i].clone()

                global_best_index = torch.argmax(personal_best_scores)
                current_best_score = personal_best_scores[global_best_index].item()

                if current_best_score > global_best_score:
                    global_best_score = current_best_score
                    global_best_position = personal_best_positions[global_best_index].clone()
                    last_improvement = iteration 

                writer.writerow([iteration, global_best_score])
                self.convergence_history.append(global_best_score)

                print(f"Iteration {iteration}: Best fitness = {global_best_score:.3f}")

                if iteration - last_improvement > 50:
                    print(f"Stagnacja wykryta w iteracji {iteration}, reset 20% cząstek.")
                    reset_count = int(0.2 * self.swarm_size)
                    indices_to_reset = random.sample(range(self.swarm_size), reset_count)
                    for idx in indices_to_reset:
                        swarm[idx] = torch.randint(0, 4, (solution_length,), dtype=torch.int)
                        velocities[idx] = torch.zeros(solution_length, dtype=torch.float)
                    
                    self.inertia = min(0.9, self.inertia + 0.1)
                    last_improvement = iteration

                if iteration % self.checkpoint_interval == 0 and iteration > 0:
                    ckpt_path = f"{self.name}_checkpoint_iter{iteration}.pt"
                    self.save(ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")

                if global_best_score >= 50.0:
                    print("Solution found by PSO!")
                    break

        self.best_solution = global_best_position.clone()

        os.makedirs(os.path.join(self.log_dir, "pso"), exist_ok=True)
        self.save_plot(
            x=range(len(self.convergence_history)),
            y=self.convergence_history,
            xlabel="Iteration",
            ylabel="Best Fitness",
            title="Convergence of PSO Algorithm",
            filename=f"pso/{self.name}.png"
        )

        metrics = self.calculate_metrics(self.convergence_history)

        metrics_dir = os.path.join("metrics/pso")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f"{self.name}.txt")
        self.log_to_file(f"Metrics: {metrics}", filename=metrics_file)

        return self.best_solution, metrics

    def solve(self, env, max_steps=None):
        if self.best_solution is None:
            raise ValueError("Brak najlepszej cząstki! Najpierw wytrenuj solver lub wczytaj model.")

        path = []
        env.reset()
        for idx, action in enumerate(self.best_solution):
            if max_steps is not None and idx >= max_steps:
                break
            state, reward, done = env.step(action.item())
            path.append(state)
            if done:
                print("Goal reached by PSOSolver!")
                break
        return path

    def solve_with_individual(self, env, individual, max_steps=None):
        path = []
        env.reset()
        for idx, action in enumerate(individual):
            if max_steps is not None and idx >= max_steps:
                break
            state, reward, done = env.step(action.item())
            path.append(state)
            if done:
                print("Goal reached by PSOSolver!")
                break
        return path

    def save(self, file_path):
        torch.save({
            'best_solution': self.best_solution,
            'swarm_size': self.swarm_size,
            'max_iterations': self.max_iterations,
            'inertia': self.inertia,
            'cognitive_coeff': self.cognitive_coeff,
            'social_coeff': self.social_coeff,
            'convergence_history': self.convergence_history
        }, file_path)

    @staticmethod
    def load(file_path):
        checkpoint = torch.load(file_path)
        solver = PSOSolver(
            swarm_size=checkpoint['swarm_size'],
            max_iterations=checkpoint['max_iterations'],
            inertia=checkpoint['inertia'],
            cognitive_coeff=checkpoint['cognitive_coeff'],
            social_coeff=checkpoint['social_coeff'],
            name="loaded_pso"
        )
        solver.best_solution = checkpoint['best_solution']
        solver.convergence_history = checkpoint.get('convergence_history', [])
        return solver



# ==============================================================================
#                            SOLVERS REGISTRY
# ==============================================================================

SOLVERS = {}

def register_solver(name, solver_cls):
    SOLVERS[name] = solver_cls
