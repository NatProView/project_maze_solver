async function postData(url = '', data = {}) {
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    return response.json();
}

function drawGrid(grid) {
    const canvas = document.getElementById("mazeCanvas");
    const ctx = canvas.getContext("2d");

    const rows = grid.length;
    const cols = grid[0].length;
    const cellWidth = canvas.width / cols;
    const cellHeight = canvas.height / rows;


    ctx.clearRect(0, 0, canvas.width, canvas.height);

    grid.forEach((row, r) => {
        row.forEach((cell, c) => {
            if (cell === 1) {
                ctx.fillStyle = "black";
            } else if (cell === 2) {
                ctx.fillStyle = "red";
            } else {
                ctx.fillStyle = "white"; 
            }
            ctx.fillRect(c * cellWidth, r * cellHeight, cellWidth, cellHeight);

            ctx.strokeStyle = "gray";
            ctx.strokeRect(c * cellWidth, r * cellHeight, cellWidth, cellHeight);
        });
    });
}

document.getElementById("solver-type").addEventListener("change", async (event) => {
    const solverType = event.target.value;
    const solverDropdown = document.getElementById("solver-name");
    const modelTypeDropdown = document.getElementById("model-type-container");

    solverDropdown.innerHTML = '<option value="">-- Select --</option>';
    modelTypeDropdown.style.display = "none"; 

    if (solverType === "ai") {
        modelTypeDropdown.style.display = "block";
        document.getElementById("model-type").addEventListener("change", async (event) => {
            const modelType = event.target.value;
            if (modelType) {
                const response = await fetch(`/get-models?type=${modelType}`);
                const data = await response.json();

                if (data.models) {
                    solverDropdown.innerHTML = data.models
                        .map((model) => `<option value="${model}">${model}</option>`)
                        .join("");
                } else {
                    solverDropdown.innerHTML = '<option value="">-- No Models Found --</option>';
                }
            }
        });
    } else if (solverType === "not-ai") {
        solverDropdown.innerHTML = `
            <option value="bfs">BFS</option>
            <option value="a_star">A*</option>
        `;
    }
});


document.getElementById("solve-maze-form").onsubmit = async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const data = Object.fromEntries(form.entries());

    const response = await postData("/solve-maze", data);

    if (response.path) {
        const mazeResponse = await postData("/get-maze", { name: data.maze_name });
        if (mazeResponse.maze) {
            const gridWithPath = mergePathIntoMaze(mazeResponse.maze, response.path);
            drawGrid(gridWithPath); 
        }
    }

    alert(response.message);
};

function mergePathIntoMaze(maze, path) {
    const mazeWithPath = maze.map(row => [...row]);

    path.forEach(([r, c]) => {
        mazeWithPath[r][c] = 2; 
    });

    return mazeWithPath;
}


document.getElementById("generate-maze-form").onsubmit = async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const data = Object.fromEntries(form.entries());
    const response = await postData("/generate-maze", data);

    if (response.maze) {
        drawGrid(response.maze);
    }

    alert(response.message);
};


const modelTypeElement = document.getElementById("model_type");
const modelParametersDiv = document.getElementById("model-parameters");

modelTypeElement.addEventListener("change", updateModelParameters);

function updateModelParameters() {
    const modelType = modelTypeElement.value;
    modelParametersDiv.innerHTML = ""; // Clear previous parameters

    if (modelType === "dqn") {
        modelParametersDiv.innerHTML = `
            <label for="num_episodes">Number of Episodes:</label>
            <input type="number" id="num_episodes" name="parameters[num_episodes]" value="500" required>
            <br>
            <label for="batch_size">Batch Size:</label>
            <input type="number" id="batch_size" name="parameters[batch_size]" value="64" required>
            <br>
            <label for="learning_rate">Learning Rate:</label>
            <input type="number" step="0.0001" id="learning_rate" name="parameters[learning_rate]" value="0.001" required>
            <br>
            <label for="gamma">Discount Factor (Gamma):</label>
            <input type="number" step="0.01" id="gamma" name="parameters[gamma]" value="0.99" required>
            <br>
            <label for="epsilon_decay">Epsilon Decay:</label>
            <input type="number" step="0.001" id="epsilon_decay" name="parameters[epsilon_decay]" value="0.995" required>
            <br>
        `;
    } else if (modelType === "genetic") {
        modelParametersDiv.innerHTML = `
            <label for="population_size">Population Size:</label>
            <input type="number" id="population_size" name="parameters[population_size]" value="100" required>
            <br>
            <label for="generations">Generations:</label>
            <input type="number" id="generations" name="parameters[generations]" value="500" required>
            <br>
            <label for="mutation_rate">Mutation Rate:</label>
            <input type="number" step="0.01" id="mutation_rate" name="parameters[mutation_rate]" value="0.1" required>
            <br>
            <label for="crossover_rate">Crossover Rate:</label>
            <input type="number" step="0.01" id="crossover_rate" name="parameters[crossover_rate]" value="0.2" required>
            <br>
            <label for="elite_fraction">Elite Fraction:</label>
            <input type="number" step="0.01" id="elite_fraction" name="parameters[elite_fraction]" value="0.05" required>
            <br>
        `;
    } else if (modelType === "pso") {
        modelParametersDiv.innerHTML = `
            <label for="swarm_size">Swarm Size:</label>
            <input type="number" id="swarm_size" name="parameters[swarm_size]" value="50" required>
            <br>
            <label for="max_iterations">Max Iterations:</label>
            <input type="number" id="max_iterations" name="parameters[max_iterations]" value="500" required>
            <br>
            <label for="inertia">Inertia:</label>
            <input type="number" step="0.01" id="inertia" name="parameters[inertia]" value="0.5" required>
            <br>
            <label for="cognitive_coeff">Cognitive Coefficient:</label>
            <input type="number" step="0.1" id="cognitive_coeff" name="parameters[cognitive_coeff]" value="1.5" required>
            <br>
            <label for="social_coeff">Social Coefficient:</label>
            <input type="number" step="0.1" id="social_coeff" name="parameters[social_coeff]" value="1.5" required>
            <br>
        `;
    }
}

updateModelParameters();

function mergePathIntoMaze(maze, path) {
    const mazeWithPath = maze.map(row => [...row]);

    path.forEach(([r, c]) => {
        mazeWithPath[r][c] = 2; 
    });

    return mazeWithPath;
}

document.getElementById("train-model-form").onsubmit = async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const data = Object.fromEntries(form.entries());

    const parameters = {};
    for (const [key, value] of form.entries()) {
        if (key.startsWith("parameters[")) {
            const paramKey = key.match(/\[([^\]]+)\]/)[1];
            parameters[paramKey] = value;
        } else {
            data[key] = value;
        }
    }
    data.parameters = parameters;

    const response = await postData("/train-model", data);

    if (response.message) {
        alert(response.message);
        
        // Update the maze visualization after training
        if (response.maze && response.path) {
            console.log("Received path:", response.path);
            const gridWithPath = mergePathIntoMaze(response.maze, response.path); // Merge the path into the maze
            drawGrid(gridWithPath); // Draw the maze with the path
        }
        // Fetch and display the plot
        const plotUrl = `/get-plot?model_type=${data.model_type}&model_name=${data.model_name}`;
        const plotImage = document.getElementById("convergence-plot");
        plotImage.src = plotUrl;
        plotImage.style.display = "block"; // Ensure the image is visible

        // Display metrics
        const metrics = response.metrics;
        if (metrics) {
            document.getElementById("metrics-mean").textContent = metrics.mean.toFixed(3);
            document.getElementById("metrics-std-dev").textContent = metrics.std_dev.toFixed(3);
            document.getElementById("metrics-skewness").textContent = metrics.skewness.toFixed(3);
            document.getElementById("training-metrics").style.display = "block"; // Show the metrics section
        }
    } else {
        alert(response.error);
    }
};