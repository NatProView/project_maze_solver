import numpy as np
import torch
from torch import nn, optim
from abc import ABC, abstractmethod
from collections import deque
import random
import matplotlib.pyplot as plt
import logging

class MazeEnv:
    def __init__(self, grid):
        self.grid = grid
        self.n_rows, self.n_cols = len(grid), len(grid[0])
        self.start = (0, 0)
        self.goal = (self.n_rows - 1, self.n_cols - 1)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Left, Up, Right, Down
        next_pos = (self.agent_pos[0] + moves[action][0], self.agent_pos[1] + moves[action][1])

        if 0 <= next_pos[0] < self.n_rows and 0 <= next_pos[1] < self.n_cols and self.grid[next_pos[0]][next_pos[1]] == 0:
            self.agent_pos = next_pos

        reward = 1 if self.agent_pos == self.goal else -0.1
        done = self.agent_pos == self.goal
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


class Solver(ABC):
    @abstractmethod
    def train(self, env):
        pass

    @abstractmethod
    def solve(self, env):
        pass

class GeneticSolver(Solver):
    def __init__(self, population_size=100, generations=500, mutation_rate=0.1, crossover_rate=0.2, elite_fraction=0.05, checkpoint_interval=250):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction  # Proporcja najlepszych osobników do zachowania
        self.checkpoint_interval = checkpoint_interval
        self.convergence_history = []  # Historia najlepszych fitness w każdej generacji

    def generate_population(self, solution_length):
        """Generuje początkową populację losowych osobników."""
        return torch.randint(0, 4, (self.population_size, solution_length), dtype=torch.int)

    def evaluate_population(self, env, population):
        """Ocena całej populacji w danym środowisku."""
        fitness_scores = torch.zeros(self.population_size, dtype=torch.float)
        for i, individual in enumerate(population):
            fitness_scores[i] = self.evaluate_individual(env, individual)
        return fitness_scores

    def evaluate_individual(self, env, individual):
        """Ocena pojedynczego osobnika."""
        env.reset()
        total_reward = 0.0
        for action in individual:
            _, reward, done = env.step(action.item())
            total_reward += reward
            if done:
                break
        return total_reward

    def crossover(self, parent1, parent2):
        """Krzyżowanie jednopunktowe."""
        if random.random() > self.crossover_rate:
            return parent1.clone()
        cut = random.randint(1, len(parent1) - 1)
        return torch.cat((parent1[:cut], parent2[cut:]))

    def mutate(self, individual, mutation_rate):
        """Mutacja z użyciem losowej maski."""
        mutation_mask = torch.rand(len(individual)) < mutation_rate
        random_mutations = torch.randint(0, 4, (len(individual),), dtype=individual.dtype)
        individual[mutation_mask] = random_mutations[mutation_mask]
        return individual

    def train(self, env):
        """Trening algorytmu genetycznego."""
        solution_length = env.n_rows * env.n_cols
        population = self.generate_population(solution_length)
        num_elites = max(1, int(self.elite_fraction * self.population_size))

        for generation in range(self.generations):
            # Ewaluacja populacji
            fitness_scores = self.evaluate_population(env, population)
            ranked_indices = torch.argsort(fitness_scores, descending=True)
            ranked_population = population[ranked_indices]

            # Zachowanie elity
            elites = ranked_population[:num_elites]

            # Dynamiczne dostosowanie mutacji
            mutation_rate = max(0.01, self.mutation_rate * (1 - generation / self.generations))

            # Selekcja rodziców
            parents = self.tournament_selection(ranked_population, fitness_scores)

            # Krzyżowanie i mutacja
            new_population = []
            for _ in range(self.population_size - num_elites):
                parent1 = parents[random.randint(0, len(parents) - 1)]
                parent2 = parents[random.randint(0, len(parents) - 1)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate)
                new_population.append(child)

            # Scalanie elity i nowej populacji
            population = torch.vstack([elites] + new_population)

            # Zapis najlepszych wyników do historii
            best_fitness = fitness_scores[ranked_indices[0]].item()
            self.convergence_history.append(best_fitness)

            # Wyświetlanie postępu
            print(f"Generation {generation}: Best fitness = {best_fitness:.3f}, Mutation rate = {mutation_rate:.3f}")

            # Sprawdzenie rozwiązania
            if best_fitness >= 15.0:
                print("Solution found!")
                return ranked_population[0]

            # Checkpoint
            if generation % self.checkpoint_interval == 0 and generation > 0:
                print(f"Checkpoint at generation {generation}: Saving state.")
                self.save(f"checkpoint_gen{generation}.pt", ranked_population[0])

        print("No solution found.")
        return ranked_population[0]

    def solve(self, env):
        """Znajdź rozwiązanie i zwróć ścieżkę na podstawie wytrenowanego modelu."""
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
        """Rozwiąż labirynt przy użyciu konkretnego osobnika."""
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
        """Zapisz najlepszego osobnika i parametry modelu."""
        torch.save({
            'best_individual': best_individual,
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
        }, file_path)

    @staticmethod
    def load(file_path):
        """Wczytaj model i najlepszego osobnika."""
        checkpoint = torch.load(file_path)
        solver = GeneticSolver(
            population_size=checkpoint['population_size'],
            generations=checkpoint['generations'],
            mutation_rate=checkpoint['mutation_rate']
        )
        best_individual = checkpoint['best_individual']
        return solver, best_individual

    def visualize_convergence(self):
        """Wizualizuj proces konwergencji algorytmu."""
        plt.plot(self.convergence_history)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence of Genetic Algorithm")
        plt.show()

    

class DQNSolver(Solver):
    def __init__(self, state_size=2, action_size=4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
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

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def save(self, file_path):
        """Save the model and metadata to a file."""
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
        """Load the model and metadata from a file."""
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

                if done:
                    break

                if len(self.replay_buffer) >= batch_size:
                    self._replay(optimizer, batch_size)

            if episode % 10 == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print(f"Episode {episode}, Total Reward: {total_reward:.3f}, Epsilon: {self.epsilon:.2f}")
        return self.dqn
            

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
    
class PSOSolver(Solver):
    def __init__(self, swarm_size=50, max_iterations=500, inertia=0.5, cognitive_coeff=1.5, social_coeff=1.5, checkpoint_interval=50):
        """
        Particle Swarm Optimization Solver.
        :param swarm_size: Liczba cząstek w roju.
        :param max_iterations: Maksymalna liczba iteracji.
        :param inertia: Współczynnik bezwładności.
        :param cognitive_coeff: Współczynnik komponentu kognitywnego.
        :param social_coeff: Współczynnik komponentu społecznego.
        :param checkpoint_interval: Co ile iteracji zapisać checkpoint.
        """
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.checkpoint_interval = checkpoint_interval
        self.convergence_history = []  # Historia najlepszych fitness w każdej iteracji

    def initialize_swarm(self, solution_length):
        """
        Inicjalizacja roju z losowymi pozycjami i prędkościami.
        """
        positions = torch.randint(0, 4, (self.swarm_size, solution_length), dtype=torch.int)
        # velocities = torch.zeros_like(positions, dtype=torch.float)
        velocities = torch.randn_like(positions, dtype=torch.float) * 2.0  # Większa początkowa losowość

        return positions, velocities

    def evaluate_particle(self, env, particle):
        """
        Ocena fitness pojedynczej cząstki.
        """
        env.reset()
        total_reward = 0
        for action in particle:
            _, reward, done = env.step(action.item())
            total_reward += reward
            if done:
                break
        return total_reward

    def evaluate_swarm(self, env, swarm):
        """
        Ocena fitness wszystkich cząstek w roju.
        """
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
            # Monitor diversity
            diversity = swarm.float().std().item()

            # If swarm stagnates, add perturbations
            if diversity < 0.5:
                print(f"Swarm is stagnating at iteration {iteration}. Introducing perturbations.")
                perturbation = torch.randint(0, 4, swarm.shape, dtype=torch.int)
                swarm = torch.where(torch.rand(swarm.shape) < 0.1, perturbation, swarm)

            # Update velocities and positions
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

                # Update personal bests
                fitness = self.evaluate_particle(env, swarm[i])
                if fitness > personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = swarm[i].clone()
                    stagnation_counter[i] = 0
                else:
                    stagnation_counter[i] += 1

            # Reset stagnant particles
            reset_indices = stagnation_counter > 20
            if reset_indices.any():
                swarm[reset_indices] = torch.randint(0, 4, (reset_indices.sum(), solution_length), dtype=torch.int)
                velocities[reset_indices] = torch.zeros_like(swarm[reset_indices], dtype=torch.float)
                stagnation_counter[reset_indices] = 0

            # Update global best
            global_best_index = torch.argmax(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index].clone()
            global_best_score = personal_best_scores[global_best_index].item()

            print(f"Iteration {iteration}: Best fitness = {global_best_score:.3f}, Diversity = {diversity:.3f}")

            if global_best_score >= 2.0:
                print("Solution found!")
                break

        return global_best_position

    def solve(self, env):
        """
        Rozwiązywanie problemu optymalizacyjnego za pomocą PSO.
        """
        best_solution = self.train(env)
        path = self.build_path(env, best_solution)
        print("Path found by PSO:", path)
        return path

    def solve_with_individual(self, env, individual):
        """
        Rozwiąż labirynt za pomocą dostarczonej cząstki (osobnika).
        """
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
        """
        Odtwórz ścieżkę rozwiązania dla najlepszego osobnika.
        """
        path = []
        env.reset()
        for action in solution:
            state, _, done = env.step(action.item())
            path.append(state)
            if done:
                break
        return path

    def save(self, file_path, best_individual):
        """Zapisz najlepszą cząstkę i parametry modelu."""
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
        """Wczytaj model PSO."""
        checkpoint = torch.load(file_path)
        return PSOSolver(
            swarm_size=checkpoint['swarm_size'],
            max_iterations=checkpoint['max_iterations'],
            inertia=checkpoint['inertia'],
            cognitive_coeff=checkpoint['cognitive_coeff'],
            social_coeff=checkpoint['social_coeff'],
        )

    def visualize_convergence(self):
        """Wizualizuj proces konwergencji algorytmu."""
        plt.plot(self.convergence_history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title("Convergence of PSO Algorithm")
        plt.show()

        
        

    
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