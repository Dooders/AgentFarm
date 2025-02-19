// Track active simulations
let activeSimulations = {};

const API_BASE_URL = 'http://192.168.1.182:8000';  // Update this to match your server IP

async function startSimulation() {
    const name = document.getElementById('simulation-name').value;
    const steps = parseInt(document.getElementById('simulation-steps').value);
    
    if (!name) {
        showError('Please enter a simulation name');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/simulation/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ name, steps })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        activeSimulations[name] = data;
        updateSimulationList();
        startPollingStatus(name);
        
    } catch (error) {
        showError(`Failed to start simulation: ${error.message}`);
    }
}

async function stopSimulation(name) {
    try {
        const response = await fetch(`${API_BASE_URL}/simulation/${name}/stop`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        delete activeSimulations[name];
        updateSimulationList();
        
    } catch (error) {
        showError(`Failed to stop simulation: ${error.message}`);
    }
}

function updateSimulationList() {
    const container = document.getElementById('simulation-cards');
    container.innerHTML = '';
    
    Object.entries(activeSimulations).forEach(([name, sim]) => {
        const progress = (sim.steps / sim.total_steps) * 100;
        
        const card = document.createElement('div');
        card.className = 'simulation-card';
        card.innerHTML = `
            <div class="simulation-info">
                <h3>${name}</h3>
                <p>Steps: ${sim.steps}/${sim.total_steps}</p>
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: ${progress}%"></div>
                </div>
            </div>
            <div class="simulation-controls">
                <button onclick="stopSimulation('${name}')" class="btn danger">Stop</button>
            </div>
        `;
        
        container.appendChild(card);
    });
}

async function pollSimulationStatus(name) {
    try {
        const response = await fetch(`${API_BASE_URL}/simulation/status/${name}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        activeSimulations[name] = data;
        updateSimulationList();
        
        if (data.running) {
            setTimeout(() => pollSimulationStatus(name), 1000);
        }
    } catch (error) {
        console.error(`Error polling simulation status: ${error.message}`);
    }
}

function startPollingStatus(name) {
    pollSimulationStatus(name);
}

function showError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="error">
            <h2>Error</h2>
            <p>${message}</p>
        </div>
    `;
}

async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Backend connection successful:', data);
        document.getElementById('connection-status').className = 'status connected';
        document.getElementById('connection-status').textContent = 'Connected to backend';
    } catch (error) {
        console.error('Backend connection failed:', error);
        document.getElementById('connection-status').className = 'status disconnected';
        document.getElementById('connection-status').textContent = 'Failed to connect to backend';
        
        // Show more detailed error in results
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = `
            <div class="error">
                <h2>Backend Connection Error</h2>
                <p>Failed to connect to backend server: ${error.message}</p>
                <p>Please ensure the backend server is running on ${API_BASE_URL}</p>
                <p>You can start it using:</p>
                <pre>uvicorn server.backend.app.main:app --reload --port 8000</pre>
            </div>
        `;
    }
}

// Initialize
window.addEventListener('load', () => {
    checkBackendConnection();
    updateSimulationList();
}); 