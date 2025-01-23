from flask import Flask, render_template, request, jsonify, send_file
import torch
import os
import csv
import time
import pandas as pd

from maze_generating import generate_maze, load_maze, save_maze
from maze_solving import bfs, a_star

from models import *

register_solver("dqn", DQNSolver)
register_solver("genetic", GeneticSolver)
register_solver("pso", PSOSolver)
register_solver("double_dqn", DoubleDQNSolver)

app = Flask(__name__)

def log_results_to_csv(file_path, headers, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get-maze", methods=["POST"])
def get_maze():
    data = request.json
    maze_name = data.get("name")
    
    if not maze_name:
        return jsonify({"error": "Maze name is required!"}), 400

    try:
        maze = load_maze(f"mazes/{maze_name}.pt")
    except FileNotFoundError:
        return jsonify({"error": f"Maze '{maze_name}' not found!"}), 404

    return jsonify({"maze": maze.tolist()})

@app.route("/get-models", methods=["GET"])
def get_models():
    model_type = request.args.get("type")
    if not model_type:
        return jsonify({"error": "Model type is required!"}), 400

    base_dir = os.path.join("trained_models", model_type)
    if not os.path.exists(base_dir):
        return jsonify({"models": []}) 

    models = [f for f in os.listdir(base_dir) if f.endswith(".pt")]
    return jsonify({"models": models})

@app.route("/generate-maze", methods=["POST"])
def generate_maze_api():
    data = request.json
    width = int(data.get("width"))
    height = int(data.get("height"))
    name = data.get("name", "maze")
    maze_path = "mazes"
    os.makedirs(os.path.dirname("mazes/"), exist_ok=True)

    maze = generate_maze(width, height)
    maze_path = f"mazes/{name}.pt"
    torch.save(maze, maze_path)

    return jsonify({"message": f"Maze '{name}' generated successfully!", "maze": maze.tolist()})

@app.route("/design-maze", methods=["POST"])
def design_maze():
    data = request.json
    name = data.get("name", "custom_maze")
    grid = data.get("grid")

    maze = torch.tensor(grid, dtype=torch.float32)
    save_maze(maze, f"mazes/{name}.pt")
    return jsonify({"message": f"Custom maze '{name}' saved successfully!"})

@app.route("/solve-maze", methods=["POST"])
def solve_maze():
    data = request.json
    maze_name = data.get("maze_name")
    solver_type = data.get("solver_type")
    solver_name = data.get("solver_name")
    model_type = data.get("model_type")

    maze_path = os.path.join("mazes", f"{maze_name}.pt")
    
    max_attempts = 100
    final_path = None
    found_solution = False
    
    
    if not os.path.exists(maze_path):
        return jsonify({"error": f"Maze '{maze_name}' not found!"}), 404
    maze = load_maze(maze_path)
    maze_size = len(maze)

    solving_time = None
    path = None
    start_time = time.time()
    used_model_type = "placeholder"
    if solver_type == "not-ai":
        if solver_name == "bfs":
            used_model_type = "bfs" 
            path = bfs(maze)
        elif solver_name == "a_star":
            path = a_star(maze)
            used_model_type = "a_star"
        else:
            return jsonify({"error": f"Solver '{solver_name}' is not recognized!"}), 400

    elif solver_type == "ai":
        model_path = os.path.join("trained_models", model_type, solver_name)
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{solver_name}' not found in '{model_type}'!"}), 404
        used_model_type = model_type  
        if model_type == "dqn":

            env = MazeEnvDQN(maze.tolist())
            solver = DQNSolver.load(model_path)
            path = solver.solve(env)
            
        elif model_type == "double_dqn":
            env = MazeEnvDQN(maze.tolist())
            solver = DoubleDQNSolver.load(model_path)
            path = solver.solve(env)
            
        elif model_type == "genetic":
            env = MazeEnvGenetic(maze.tolist())
            solver, best_individual = GeneticSolver.load(model_path)
            path = solver.solve_with_individual(env, best_individual)

        elif model_type == "pso":
            env = MazeEnvPSO(maze.tolist())
            solver = PSOSolver.load(model_path)
            path = solver.solve(env)

        else:
            return jsonify({"error": f"Model type '{model_type}' is not recognized!"}), 400
    else:
        return jsonify({"error": "Invalid solver type!"}), 400
    
    solving_time = time.time() - start_time
    path_length = len(path) if path else 0

    log_results_to_csv(
        "solving_logs.csv",
        ["Algorithm type", "Model", "Maze Size", "Solving Time (s)", "Path Length"],
        [solver_type, used_model_type, maze_size, round(solving_time, 2), path_length]
    )
    
    return jsonify({
        "message": f"Maze '{maze_name}' solved using '{solver_type}' solver '{solver_name}'!",
        "path": path
    })

@app.route("/list-solvers", methods=["GET"])
def list_solvers():
    return jsonify({"solvers": list(SOLVERS.keys())})

@app.route("/train-model", methods=["POST"])
def train_model():
    data = request.json
    model_type = data.get("model_type")
    model_name = data.get("model_name")
    maze_name = data.get("maze_name")
    parameters = data.get("parameters", {})

    if not (model_type and model_name and maze_name):
        return jsonify({"error": "model_type, model_name, and maze_name are required"}), 400

    maze_path = f"mazes/{maze_name}.pt"
    if not os.path.exists(maze_path):
        return jsonify({"error": f"Maze '{maze_name}' not found!"}), 404

    maze = load_maze(maze_path)
    maze_size = len(maze)
    training_time = None
    path = None
    metrics = {}

    start_time = time.time()

    if model_type == "genetic":
        env = MazeEnvGenetic(maze.tolist())

        population_size = int(parameters.get("population_size", 100))
        generations = int(parameters.get("generations", 500))
        mutation_rate = float(parameters.get("mutation_rate", 0.1))
        crossover_rate = float(parameters.get("crossover_rate", 0.2))
        elite_fraction = float(parameters.get("elite_fraction", 0.05))

        solver = GeneticSolver(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_fraction=elite_fraction,
            name=model_name
        )
        best_individual, metrics = solver.train(env)
        training_time = time.time() - start_time

        save_path = f"trained_models/genetic/{model_name}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        solver.save(save_path)

        path = solver.solve_with_individual(env, best_individual)

    elif model_type == "dqn":
        env = MazeEnvDQN(maze.tolist())

        num_episodes = int(parameters.get("num_episodes", 500))
        batch_size = int(parameters.get("batch_size", 64))
        learning_rate = float(parameters.get("learning_rate", 0.001))
        gamma = float(parameters.get("gamma", 0.99))
        epsilon_decay = float(parameters.get("epsilon_decay", 0.995))
        epsilon_min = float(parameters.get("epsilon_min", 0.01))
        replay_buffer_size = int(parameters.get("replay_buffer_size", 100000))
        max_steps = int(parameters.get("max_steps", 100))
    
        solver = DQNSolver(
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            learning_rate=learning_rate,
            replay_buffer_size=replay_buffer_size,
            name=model_name
        )
        _, metrics = solver.train(env, num_episodes=num_episodes, batch_size=batch_size, max_steps=max_steps)
        training_time = time.time() - start_time

        save_path = f"trained_models/dqn/{model_name}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        solver.save(save_path)

        path = solver.solve(env)
        
    elif model_type == "double_dqn":
        env = MazeEnvDQN(maze.tolist())

        num_episodes = int(parameters.get("num_episodes", 500))
        batch_size = int(parameters.get("batch_size", 64))
        learning_rate = float(parameters.get("learning_rate", 0.0005))
        gamma = float(parameters.get("gamma", 0.99))
        epsilon_decay = float(parameters.get("epsilon_decay", 0.995))
        epsilon_min = float(parameters.get("epsilon_min", 0.01))
        replay_buffer_size = int(parameters.get("replay_buffer_size", 100000))
        tau = float(parameters.get("tau", 0.01))
        max_steps = int(parameters.get("max_steps", 100))

        solver = DoubleDQNSolver(
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            learning_rate=learning_rate,
            replay_buffer_size=replay_buffer_size,
            tau=tau,
            name=model_name
        )

        start_time = time.time()
        _, metrics = solver.train(env, num_episodes=num_episodes, batch_size=batch_size)
        training_time = time.time() - start_time

        save_path = f"trained_models/double_dqn/{model_name}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        solver.save(save_path)

        path = solver.solve(env)

        path_length = len(path)
        log_results_to_csv(
            "training_logs.csv",
            ["Algorithm", "Maze Size", "Training Time (s)", "Path Length"],
            [model_type, maze_size, round(training_time, 2), path_length]
        )
        return jsonify({
            "message": f"Double DQN model '{model_name}' trained successfully!",
            "path": path,
            "maze": maze.tolist(),
            "metrics": metrics
        })

    elif model_type == "pso":
        env = MazeEnvPSO(maze.tolist())

        swarm_size = int(parameters.get("swarm_size", 50))
        max_iterations = int(parameters.get("max_iterations", 500))
        inertia = float(parameters.get("inertia", 0.5))
        cognitive_coeff = float(parameters.get("cognitive_coeff", 1.5))
        social_coeff = float(parameters.get("social_coeff", 1.5))

        solver = PSOSolver(
            swarm_size=swarm_size,
            max_iterations=max_iterations,
            inertia=inertia,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            name=model_name
        )
        best_solution, metrics = solver.train(env)
        training_time = time.time() - start_time

        save_path = f"trained_models/pso/{model_name}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        solver.save(save_path)

        path = solver.solve(env)

    else:
        return jsonify({"error": "Invalid model type or parameters!"}), 400

    path_length = len(path) if path else 0

    log_results_to_csv(
        "training_logs.csv",
        ["Algorithm", "Maze Size", "Training Time (s)", "Path Length"],
        [model_type, maze_size, round(training_time, 2), path_length]
    )
    print(f"Training log: {[model_type, maze_size, round(training_time, 2), path_length]}")

    return jsonify({
        "message": f"Model '{model_name}' of type '{model_type}' trained successfully!",
        "path": path,
        "maze": maze.tolist(),
        "metrics": metrics
    })

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analytics-data", methods=["GET"])
def analytics_data():
    solving_logs_path = "solving_logs.csv"
    training_logs_path = "training_logs.csv"

    solving_data = []
    training_data = []

    if os.path.exists(solving_logs_path):
        solving_data = pd.read_csv(solving_logs_path).to_dict(orient="records")
    if os.path.exists(training_logs_path):
        training_data = pd.read_csv(training_logs_path).to_dict(orient="records")

    return jsonify({
        "solving_logs": solving_data,
        "training_logs": training_data
    })

@app.route("/get-plot", methods=["GET"])
def get_plot():
    model_type = request.args.get("model_type")
    model_name = request.args.get("model_name")

    if not model_type or not model_name:
        return jsonify({"error": "Model type and name are required!"}), 400

    plot_path = os.path.join("logs", model_type, f"{model_name}.png")
    print(f"Plot Path: {plot_path}")
    
    if not os.path.exists(plot_path):
        return jsonify({"error": f"Plot for model '{model_name}' not found!"}), 404

    return send_file(plot_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
