async function fetchAnalyticsData() {
    const response = await fetch("/analytics-data");
    return response.json();
}

let chartInstance = null;
function createChart(ctx, chartLabel, dataset) {
    if (chartInstance) {
        chartInstance.destroy();
    }
    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dataset.labels,
            datasets: [{
                label: chartLabel,
                data: dataset.data,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}


function getUniqueMazeSizes(solvingLogs, trainingLogs) {
    const sizesSolving = solvingLogs.map(log => log["Maze Size"]);
    const sizesTraining = trainingLogs.map(log => log["Maze Size"] || 0);
    const allSizes = [...sizesSolving, ...sizesTraining];
    const unique = [...new Set(allSizes)];
    unique.sort((a, b) => a - b);
    return unique;
}

function populateMazeSizeSelect(uniqueSizes) {
    const select = document.getElementById("maze-size-select");
    select.innerHTML = "<option value=''>-- All Sizes --</option>";
    uniqueSizes.forEach(sz => {
        const opt = document.createElement("option");
        opt.value = sz;
        opt.textContent = sz;
        select.appendChild(opt);
    });
}

function updateChart(analyticsType, data) {
    const ctx = document.getElementById("analyticsChart").getContext("2d");

    const solvingLogs = data.solving_logs || [];
    const trainingLogs = data.training_logs || [];

    const selectedSize = document.getElementById("maze-size-select").value;

    if (analyticsType === "solving_time") {
        let filtered = solvingLogs;
        if (selectedSize !== "") {
            filtered = solvingLogs.filter(log => +log["Maze Size"] === +selectedSize);
        }
        const labels = filtered.map(log => `${log["Algorithm type"]} (${log["Model"]})`);
        const times = filtered.map(log => log["Solving Time (s)"]);
        createChart(ctx, "Solving Time (s)", { labels, data: times });

    } else if (analyticsType === "path_length") {
        let filtered = solvingLogs;
        if (selectedSize !== "") {
            filtered = solvingLogs.filter(log => +log["Maze Size"] === +selectedSize);
        }
        const labels = filtered.map(log => `${log["Algorithm type"]} (${log["Model"]})`);
        const lengths = filtered.map(log => log["Path Length"]);
        createChart(ctx, "Path Length", { labels, data: lengths });

    } else if (analyticsType === "training_time") {
        let filtered = trainingLogs;
        if (selectedSize !== "") {
            filtered = trainingLogs.filter(log => +log["Maze Size"] === +selectedSize);
        }
        const labels = filtered.map(log => log["Algorithm"] || log["Model"] || "Unknown");
        const times = filtered.map(log => log["Training Time (s)"]);
        createChart(ctx, "Training Time (s)", { labels, data: times });
    }
}

function toggleMazeSizeSelect(analyticsType) {
    const mazeSizeContainer = document.getElementById("maze-size-container");
    if (analyticsType === "solving_time" 
        || analyticsType === "path_length"
        || analyticsType === "training_time") 
    {
        mazeSizeContainer.style.display = "block";
    } else {
        mazeSizeContainer.style.display = "none";
    }
}

async function initializeDashboard() {
    const data = await fetchAnalyticsData();

    const uniqueSizes = getUniqueMazeSizes(data.solving_logs || [], data.training_logs || []);
    populateMazeSizeSelect(uniqueSizes);

    const analyticsTypeSelect = document.getElementById("analytics-type");
    const initialType = analyticsTypeSelect.value;
    toggleMazeSizeSelect(initialType);
    updateChart(initialType, data);

    analyticsTypeSelect.addEventListener("change", () => {
        const newType = analyticsTypeSelect.value;
        toggleMazeSizeSelect(newType);
        updateChart(newType, data);
    });

    document.getElementById("maze-size-select").addEventListener("change", () => {
        updateChart(analyticsTypeSelect.value, data);
    });
}

initializeDashboard();
