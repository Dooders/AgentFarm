async function runSimulation() {
    const name = document.getElementById('simulation-name').value;
    if (!name) {
        document.getElementById('results').innerHTML = `
            <h2>Error</h2>
            <p>Please enter a simulation name</p>
        `;
        return;
    }

    try {
        const response = await fetch('http://localhost:8000/simulation/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ name: name })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = `
            <h2>Error</h2>
            <p>Failed to run simulation: ${error.message}</p>
        `;
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <h2>Results</h2>
        <p>Name: ${data.name}</p>
        <p>Result: ${data.result}</p>
        <p>Timestamp: ${new Date(data.timestamp).toLocaleString()}</p>
    `;
}

async function checkBackendConnection() {
    try {
        const response = await fetch('http://localhost:8000/health');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Backend connection successful:', data);
        document.getElementById('connection-status').textContent = 'Connected to backend';
    } catch (error) {
        console.error('Backend connection failed:', error);
        document.getElementById('connection-status').textContent = 'Failed to connect to backend';
    }
}

window.addEventListener('load', checkBackendConnection); 