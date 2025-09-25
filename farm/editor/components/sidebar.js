class Sidebar {
    constructor() {
        this.element = document.getElementById('sidebar');
        this.init();
    }

    init() {
		this.element.innerHTML = `
            <div class="sidebar-header">
				<h2>Simulation Controls</h2>
				<button id="open-explorer" class="open-explorer">Config Explorer</button>
				<button id="toggle-grayscale-sidebar" class="open-explorer" title="Toggle grayscale UI">Grayscale</button>
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
		// Initialize grayscale button state from persisted preference
		try {
			const enabled = localStorage.getItem('ui:grayscale') === '1';
			if (enabled) document.body.classList.add('grayscale');
			const grayBtn = document.getElementById('toggle-grayscale-sidebar');
			if (grayBtn) grayBtn.classList.toggle('toggled', enabled);
		} catch (e) {}
    }

    attachEventListeners() {
		document.getElementById('open-explorer').addEventListener('click', () => {
			if (window.showConfigExplorer) window.showConfigExplorer();
		});
		const grayBtn = document.getElementById('toggle-grayscale-sidebar');
		if (grayBtn) {
			grayBtn.addEventListener('click', () => {
				const enabled = document.body.classList.toggle('grayscale');
				try { localStorage.setItem('ui:grayscale', enabled ? '1' : '0'); } catch (e) {}
				grayBtn.classList.toggle('toggled', enabled);
				if (typeof window !== 'undefined' && typeof window.__setExplorerGrayscale === 'function') {
					window.__setExplorerGrayscale(enabled);
				}
			});
		}
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