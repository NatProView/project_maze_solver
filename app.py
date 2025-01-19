from flask import Flask, render_template, request, jsonify
import torch
from maze_generating import *
from models import MazeEnv, SOLVERS, register_solver, DQNSolver, GeneticSolver, PSOSolver
from models_dqn import MazeEnv as DqnMazeEnv
from models_dqn import DQNSolver as new_DqnSolver
from maze_solving import *
import os, csv, time
import pandas as pd
# matplotlib requests flask pandas
# torch i inne torhcowe rzeczy

register_solver("dqn", new_DqnSolver)
register_solver("genetic", GeneticSolver)
register_solver("pso", PSOSolver)

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
    
    if not os.path.exists(maze_path):
        return jsonify({"error": f"Maze '{maze_name}' not found!"}), 404
    maze = load_maze(maze_path)
    maze_size = len(maze)
    env = MazeEnv(maze.tolist())
    solving_time = None
    path = None
    start_time = time.time()
    if solver_type == "not-ai":
        if solver_name == "bfs":
            path = bfs(maze)
        elif solver_name == "a_star":
            path = a_star(maze)
        else:
            return jsonify({"error": f"Solver '{solver_name}' is not recognized!"}), 400

    elif solver_type == "ai":
        model_path = os.path.join("trained_models", model_type, solver_name)
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{solver_name}' not found in '{model_type}'!"}), 404

        if model_type == "dqn":
            solver = new_DqnSolver.load(model_path)
            path = solver.solve(env)
        elif model_type == "genetic":
            solver, best_individual = GeneticSolver.load(model_path)
            path = solver.solve_with_individual(env, best_individual)
        elif model_type == "pso":
            solver = PSOSolver().load(model_path)
            path = solver.solve(env)
        else:
            return jsonify({"error": f"Model type '{model_type}' is not recognized!"}), 400
    else:
        return jsonify({"error": "Invalid solver type!"}), 400
    
    solving_time = time.time() - start_time
    path_length = len(path)
    log_results_to_csv(
        "solving_logs.csv",
        ["Algorithm", "Maze Size", "Solving Time (s)", "Path Length"],
        [solver_type, maze_size, round(solving_time, 2), path_length]
    )
    
    return jsonify({
        "message": f"Maze '{maze_name}' solved using '{solver_type}' solver '{solver_name}'!",
        "path": path
    })


@app.route("/list-solvers", methods=["GET"])
def list_solvers():
    return jsonify({"solvers": list(SOLVERS.keys())})



@app.route("/train-model-page")
def train_model_page():
    return render_template("train_model.html")

@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route("/train-model", methods=["POST"])
def train_model():
    data = request.json
    model_type = data.get("model_type")
    model_name = data.get("model_name")
    maze_name = data.get("maze_name")
    parameters = data.get("parameters")
    
    maze = load_maze(f"mazes/{maze_name}.pt")
    maze_size = len(maze)
    training_time = None

    if model_type == "genetic":

        env = MazeEnv(maze.tolist())

        population_size = int(parameters.get("population_size", 100))
        generations = int(parameters.get("generations", 500))
        mutation_rate = float(parameters.get("mutation_rate", 0.1))

        solver = GeneticSolver(
            population_size=population_size, generations=generations, mutation_rate=mutation_rate
        )
        start_time = time.time()
        best_individual = solver.train(env)
        training_time = time.time() - start_time
        

        solver.save(f"trained_models/genetic/{model_name}.pt", best_individual)
        path = solver.solve(env)
        
        return jsonify({
            "message": f"Genetic model '{model_name}' trained successfully!",
            "path": path,
            "maze": maze.tolist()
        })

    if model_type == "dqn":
        env = MazeEnv(maze.tolist())

        num_episodes = int(parameters.get("num_episodes", 500))
        batch_size = int(parameters.get("batch_size", 64))
        learning_rate = float(parameters.get("learning_rate", 0.001))

        solver = new_DqnSolver(learning_rate=learning_rate)
        start_time = time.time()
        trained_model = solver.train(env, num_episodes=num_episodes, batch_size=batch_size)
        training_time = time.time() - start_time
        
        solver.save(f"trained_models/dqn/{model_name}.pt")
        path = solver.solve(env)
        
        return jsonify({
            "message": f"DQN model '{model_name}' trained successfully!",
            "path": path,
            "maze": maze.tolist()
        })
    
    if model_type == "pso":
        env = MazeEnv(maze.tolist())

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
            social_coeff=social_coeff
        )
        
        start_time = time.time()
        best_individual = solver.train(env)
        training_time = time.time() - start_time
        
        solver.save(f"trained_models/pso/{model_name}.pt", best_individual)
        path = solver.solve(env)

        return jsonify({
            "message": f"PSO model '{model_name}' trained successfully!",
            "path": path,
            "maze": maze.tolist()
        })

    path_length = len(path)
    log_results_to_csv(
        "training_logs.csv",
        ["Algorithm", "Maze Size", "Training Time (s)", "Path Length"],
        [model_type, maze_size, round(training_time, 2), path_length]
    )
    print(f"Do loga: {[model_type, maze_size, round(training_time, 2), path_length]}")
    return jsonify({"error": "Invalid model type or parameters!"}), 400

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analytics-data", methods=["GET"])
def analytics_data():
    solving_logs_path = "solving_logs.csv"
    training_logs_path = "training_logs.csv"

    solving_data = pd.read_csv(solving_logs_path).to_dict(orient="records") if os.path.exists(solving_logs_path) else []
    training_data = pd.read_csv(training_logs_path).to_dict(orient="records") if os.path.exists(training_logs_path) else []

    return jsonify({
        "solving_logs": solving_data,
        "training_logs": training_data
    })

if __name__ == "__main__":
    app.run(debug=True)
