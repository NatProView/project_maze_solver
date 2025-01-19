async function fetchAnalyticsData() {
    const response = await fetch("/analytics-data");
    return response.json();
}

let chart; // To hold the Chart.js instance

function createChart(ctx, label, datasets) {
    if (chart) {
        chart.destroy(); // Destroy the previous chart instance
    }

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: datasets.labels,
            datasets: [{
                label: label,
                data: datasets.data,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
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

function updateChart(analyticsType, data) {
    const ctx = document.getElementById("analyticsChart").getContext("2d");

    if (analyticsType === "training_time") {
        const trainingLogs = data.training_logs;
        const labels = trainingLogs.map(log => `Size: ${log['Maze Size']}`);
        const trainingTimes = trainingLogs.map(log => log['Training Time (s)']);
        createChart(ctx, "Training Time (s)", { labels, data: trainingTimes });
    } else if (analyticsType === "solving_time") {
        const solvingLogs = data.solving_logs;
        const labels = solvingLogs.map(log => `${log.Algorithm} (Size: ${log['Maze Size']})`);
        const solvingTimes = solvingLogs.map(log => log['Solving Time (s)']);
        createChart(ctx, "Solving Time (s)", { labels, data: solvingTimes });
    } else if (analyticsType === "path_length") {
        const solvingLogs = data.solving_logs;
        const labels = solvingLogs.map(log => log.Algorithm);
        const pathLengths = solvingLogs.map(log => log['Path Length']);
        createChart(ctx, "Average Path Length", { labels, data: pathLengths });
    }
}

async function initializeDashboard() {
    const data = await fetchAnalyticsData();

    // Initialize with default analytics
    const analyticsType = document.getElementById("analytics-type").value;
    updateChart(analyticsType, data);

    // Change chart when a new type is selected
    document.getElementById("analytics-type").addEventListener("change", (event) => {
        updateChart(event.target.value, data);
    });
}

initializeDashboard();
