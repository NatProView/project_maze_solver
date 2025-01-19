# Maze Solver

## Overview
Maze Solver is a Flask-based web application for generating mazes, solving them using various algorithms, and training AI models to solve mazes. 

## Technologies Used
- **Backend**: Python with Flask
- **Frontend**: HTML, CSS (Bootstrap), JavaScript (with Chart.js)
- **Data Handling**: CSV files for logging analytics
- **AI models training library:** PyTorch
- **Dependencies**:
  - `torch` 
  - `pandas` 
  - `requests`
  - `tkinter`

## Features
1. **Maze Generation**:
   - Generate mazes of custom sizes using Eller's Algorithm (procedural)
   - Save and load mazes as `.pt` files.
2. **Maze Solving**:
   - Solve mazes using classical algorithms like BFS and A*.
   - Train AI models (DQN, Genetic Algorithm, PSO) to solve mazes.
   - Visualize the solving path on the maze.

3. **Model Training**:
   - Train AI models with customizable parameters.
   - Save trained models for future use.

4. **Analytics Dashboard**:
   - Visualize training and solving performance using interactive charts.
   - Analyze metrics such as solving time, training time, and path length by algorithm and maze size.


## How to Run
1. **Install Dependencies**:
   ```bash
   pip install flask torch pandas chart.js requests
   ```

2. **Start the Application**:
   ```bash
   python app.py
   ```

3. **Access the App**:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

## Key Endpoints
- **Homepage**: `/` - Interact with maze generation, solving, and training.
- **Dashboard**: `/dashboard` - Visualize analytics.
- **API Endpoints**:
  - `/generate-maze` (POST): Generate a new maze.
  - `/solve-maze` (POST): Solve a maze using specified algorithm or AI model.
  - `/train-model` (POST): Train an AI model.
  - `/analytics-data` (GET): Fetch data for analytics.
