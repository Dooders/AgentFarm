// Track active simulations
let activeSimulations = {};

const API_BASE_URL = 'http://192.168.1.182:8000';  // Update this to match your server IP

// --------------------
// Config service utils
// --------------------
async function loadConfig() {
    const pathInput = document.getElementById('config-path');
    const editor = document.getElementById('config-editor');
    const results = document.getElementById('config-results');
    try {
        const response = await fetch(`${API_BASE_URL}/config/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify({ path: pathInput && pathInput.value ? pathInput.value : null })
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            const errs = data.errors && data.errors.length ? data.errors.join('; ') : `HTTP ${response.status}`;
            results.innerHTML = `<div class="error"><strong>Load failed:</strong> ${data.message || errs}</div>`;
            return;
        }
        editor.value = JSON.stringify(data.config, null, 2);
        results.innerHTML = `<div class="success"><strong>Loaded.</strong> ${data.path ? `From: ${data.path}` : ''}</div>`;
    } catch (e) {
        results.innerHTML = `<div class="error"><strong>Load failed:</strong> ${e.message}</div>`;
    }
}

async function validateConfig() {
    const editor = document.getElementById('config-editor');
    const results = document.getElementById('config-results');
    try {
        const parsed = JSON.parse(editor.value || '{}');
        const response = await fetch(`${API_BASE_URL}/config/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify({ config: parsed })
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            const errs = data.errors && data.errors.length ? data.errors.join('; ') : `HTTP ${response.status}`;
            results.innerHTML = `<div class="error"><strong>Invalid:</strong> ${data.message || errs}</div>`;
            return;
        }
        // Normalize editor with server-canonical config
        editor.value = JSON.stringify(data.config, null, 2);
        results.innerHTML = `<div class="success"><strong>Valid.</strong> ${data.message || ''}</div>`;
    } catch (e) {
        results.innerHTML = `<div class="error"><strong>Validation failed:</strong> ${e.message}</div>`;
    }
}

async function saveConfig() {
    const pathInput = document.getElementById('config-path');
    const editor = document.getElementById('config-editor');
    const results = document.getElementById('config-results');
    try {
        const parsed = JSON.parse(editor.value || '{}');
        const response = await fetch(`${API_BASE_URL}/config/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify({ config: parsed, path: pathInput && pathInput.value ? pathInput.value : null })
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            const errs = data.errors && data.errors.length ? data.errors.join('; ') : `HTTP ${response.status}`;
            results.innerHTML = `<div class="error"><strong>Save failed:</strong> ${data.message || errs}</div>`;
            return;
        }
        // Normalize editor with saved content
        editor.value = JSON.stringify(data.config, null, 2);
        results.innerHTML = `<div class="success"><strong>Saved.</strong> ${data.path ? `To: ${data.path}` : ''}</div>`;
    } catch (e) {
        results.innerHTML = `<div class="error"><strong>Save failed:</strong> ${e.message}</div>`;
    }
}

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
        
        const data = await response.json();
        
        // Handle all stop cases as success
        if (data.status === "stopped") {
            delete activeSimulations[name];
            updateSimulationList();
            await getSimulationHistory(); // Refresh history
        }
        
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
        if (response.status === 404) {
            // Simulation is completed and cleaned up
            delete activeSimulations[name];
            updateSimulationList();
            await getSimulationHistory(); // Refresh history
            return;
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        activeSimulations[name] = data;
        updateSimulationList();
        
        if (data.running) {
            setTimeout(() => pollSimulationStatus(name), 1000);
        } else {
            // Simulation completed naturally
            await stopSimulation(name); // Clean up gracefully
            await getSimulationHistory(); // Refresh history
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

async function loadSimulation(id) {
    try {
        const response = await fetch(`${API_BASE_URL}/simulation/load/${id}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        activeSimulations[data.name] = data;
        updateSimulationList();
        startPollingStatus(data.name);
        
    } catch (error) {
        showError(`Failed to load simulation: ${error.message}`);
    }
}

async function deleteSimulation(id) {
    try {
        const response = await fetch(`${API_BASE_URL}/simulation/history/${id}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        await getSimulationHistory(); // Refresh history after deletion
        
    } catch (error) {
        showError(`Failed to delete simulation: ${error.message}`);
    }
}

async function getSimulationHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/simulation/history`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const history = await response.json();
        const container = document.getElementById('history-cards');
        container.innerHTML = '';
        
        history.forEach(sim => {
            const card = document.createElement('div');
            card.className = 'simulation-card history-card';
            const timestamp = new Date(sim.timestamp).toLocaleString();
            
            card.innerHTML = `
                <div class="simulation-info">
                    <h3>${sim.name}</h3>
                    <p>Result: ${sim.result.toFixed(2)}</p>
                    <p>Run on: ${timestamp}</p>
                </div>
                <div class="simulation-controls">
                    <button onclick="loadSimulation(${sim.id})" class="btn secondary">Load</button>
                    <button onclick="deleteSimulation(${sim.id})" class="btn danger">Delete</button>
                </div>
            `;
            
            container.appendChild(card);
        });
        
    } catch (error) {
        console.error('Failed to fetch simulation history:', error);
    }
}

async function refreshHistory() {
    await getSimulationHistory();
}

// Initialize
window.addEventListener('load', () => {
    checkBackendConnection();
    updateSimulationList();
    getSimulationHistory();
}); 