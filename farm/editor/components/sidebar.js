class Sidebar {
    constructor() {
        this.element = document.getElementById('sidebar');
        this.init();
    }

    init() {
        this.element.innerHTML = `
            <div class="sidebar-header">
                <h2>Simulation Controls</h2>
            </div>
            <div class="sidebar-content">
                <button id="new-sim">New Simulation</button>
                <button id="open-sim">Open Simulation</button>
                <button id="export-data">Export Data</button>
                <div class="config-section">
                    <h3>Configuration</h3>
                    <div class="config-item">
                        <label>Environment Width:</label>
                        <input type="number" id="env-width" value="100">
                    </div>
                    <div class="config-item">
                        <label>Environment Height:</label>
                        <input type="number" id="env-height" value="100">
                    </div>
                    <div class="config-item">
                        <label>Initial Resources:</label>
                        <input type="number" id="init-resources" value="1000">
                    </div>
                </div>
            </div>
        `;

        this.attachEventListeners();
    }

    attachEventListeners() {
        document.getElementById('new-sim').addEventListener('click', () => {
            this.createNewSimulation();
        });

        document.getElementById('open-sim').addEventListener('click', () => {
            this.openSimulation();
        });

        document.getElementById('export-data').addEventListener('click', () => {
            this.exportData();
        });
    }

    async createNewSimulation() {
        const config = {
            width: parseInt(document.getElementById('env-width').value),
            height: parseInt(document.getElementById('env-height').value),
            initialResources: parseInt(document.getElementById('init-resources').value)
        };

        try {
            const response = await fetch('http://localhost:5000/api/simulation/new', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                throw new Error('Failed to create simulation');
            }

            const data = await response.json();
            // Trigger visualization update
            window.dispatchEvent(new CustomEvent('simulation-created', { detail: data }));
        } catch (error) {
            console.error('Error creating simulation:', error);
        }
    }

    async openSimulation() {
        // Implementation for opening existing simulation
    }

    async exportData() {
        // Implementation for exporting data
    }
}

// Initialize sidebar when document is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sidebar = new Sidebar();
}); 