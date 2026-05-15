class Sidebar {
    constructor() {
        this.element = document.getElementById('sidebar');
        this.experimentPollInterval = null;
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
                    <h3>Experiment Builder</h3>
                    <div class="config-item">
                        <label>Experiment Name:</label>
                        <input type="text" id="exp-name" value="intrinsic-dashboard-run">
                    </div>
                    <div class="config-item">
                        <label>Steps:</label>
                        <input type="number" id="exp-steps" value="400" min="1">
                    </div>
                    <div class="config-item">
                        <label>Snapshot Interval:</label>
                        <input type="number" id="exp-snapshot-interval" value="25" min="1">
                    </div>
                    <div class="config-item">
                        <label>Selection Pressure:</label>
                        <select id="exp-selection-pressure">
                            <option value="none">none</option>
                            <option value="low">low</option>
                            <option value="medium" selected>medium</option>
                            <option value="high">high</option>
                        </select>
                    </div>
                    <div class="config-item">
                        <label>Enable Speciation:</label>
                        <input type="checkbox" id="exp-enable-speciation" checked>
                    </div>
                    <div class="builder-actions">
                        <button id="validate-exp">Validate Manifest</button>
                        <button id="run-exp">Run Experiment</button>
                        <button id="save-exp-preset">Save Preset</button>
                        <button id="load-exp-preset">Load Preset</button>
                        <button id="load-latest-exp">Load Latest</button>
                    </div>
                    <div id="exp-status" class="exp-status">No experiment run yet.</div>
                </div>
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
        document.getElementById('validate-exp').addEventListener('click', () => {
            this.validateExperimentManifest();
        });
        document.getElementById('run-exp').addEventListener('click', () => {
            this.runExperimentFromBuilder();
        });
        document.getElementById('save-exp-preset').addEventListener('click', () => {
            this.saveExperimentPreset();
        });
        document.getElementById('load-exp-preset').addEventListener('click', () => {
            this.loadExperimentPreset();
        });
        document.getElementById('load-latest-exp').addEventListener('click', () => {
            this.loadLatestExperimentRun();
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

    buildIntrinsicManifest() {
        const width = parseInt(document.getElementById('env-width').value, 10);
        const height = parseInt(document.getElementById('env-height').value, 10);
        const initialResources = parseInt(document.getElementById('init-resources').value, 10);
        const experimentName = document.getElementById('exp-name').value.trim() || 'intrinsic-dashboard-run';
        const steps = parseInt(document.getElementById('exp-steps').value, 10);
        const snapshotInterval = parseInt(document.getElementById('exp-snapshot-interval').value, 10);
        const selectionPressure = document.getElementById('exp-selection-pressure').value;
        const speciationEnabled = document.getElementById('exp-enable-speciation').checked;

        return {
            schema_version: 1,
            experiment_type: 'intrinsic_evolution',
            experiment_name: experimentName,
            base_simulation_config: {
                'environment.width': width,
                'environment.height': height,
                'resources.initial_resources': initialResources
            },
            experiment_config: {
                num_steps: steps,
                snapshot_interval: snapshotInterval,
                initial_conditions: {
                    profile: 'stable'
                },
                policy: {
                    selection_pressure: selectionPressure
                },
                speciation: {
                    enabled: speciationEnabled
                }
            },
            dashboard_preset: {
                default_views: ['summary_cards', 'gene_trajectories', 'population_dynamics']
            }
        };
    }

    updateExperimentStatus(message, isError = false) {
        const status = document.getElementById('exp-status');
        status.textContent = message;
        status.classList.toggle('error', isError);
    }

    async validateExperimentManifest() {
        const manifest = this.buildIntrinsicManifest();
        this.updateExperimentStatus('Validating manifest...');
        try {
            const response = await fetch('http://localhost:5000/api/experiments/manifests/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ manifest })
            });
            const data = await response.json();
            const result = data.data || {};
            if (!result.is_valid) {
                this.updateExperimentStatus(`Manifest invalid: ${(result.errors || []).join('; ')}`, true);
                return;
            }
            this.updateExperimentStatus('Manifest is valid.');
        } catch (error) {
            this.updateExperimentStatus(`Validation failed: ${error.message}`, true);
        }
    }

    async runExperimentFromBuilder() {
        const manifest = this.buildIntrinsicManifest();
        this.updateExperimentStatus('Starting experiment run...');
        try {
            const response = await fetch('http://localhost:5000/api/experiments/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ manifest })
            });
            if (!response.ok) {
                const failure = await response.text();
                throw new Error(failure || 'Failed to start experiment');
            }
            const data = await response.json();
            const runId = data.run_id;
            this.updateExperimentStatus(`Run ${runId} started.`);
            window.dispatchEvent(new CustomEvent('experiment-run-created', { detail: { run_id: runId } }));
            this.pollExperimentRun(runId);
        } catch (error) {
            this.updateExperimentStatus(`Run failed to start: ${error.message}`, true);
        }
    }

    async pollExperimentRun(runId) {
        if (this.experimentPollInterval) {
            clearInterval(this.experimentPollInterval);
        }
        this.experimentPollInterval = setInterval(async () => {
            try {
                const response = await fetch(`http://localhost:5000/api/experiments/${runId}/status`);
                const payload = await response.json();
                const status = payload?.data?.status;
                window.dispatchEvent(new CustomEvent('experiment-run-status', { detail: payload.data }));
                this.updateExperimentStatus(`Run ${runId}: ${status}`);
                if (status === 'completed') {
                    clearInterval(this.experimentPollInterval);
                    this.experimentPollInterval = null;
                    window.dispatchEvent(new CustomEvent('experiment-run-completed', { detail: { run_id: runId } }));
                }
                if (status === 'error') {
                    clearInterval(this.experimentPollInterval);
                    this.experimentPollInterval = null;
                    const message = payload?.data?.error_message || 'Unknown error';
                    this.updateExperimentStatus(`Run ${runId} errored: ${message}`, true);
                }
            } catch (error) {
                clearInterval(this.experimentPollInterval);
                this.experimentPollInterval = null;
                this.updateExperimentStatus(`Run polling failed: ${error.message}`, true);
            }
        }, 2000);
    }

    async loadLatestExperimentRun() {
        this.updateExperimentStatus('Loading latest completed run...');
        try {
            const response = await fetch('http://localhost:5000/api/experiments/runs');
            const payload = await response.json();
            const runs = payload?.data || [];
            const completed = runs.filter(run => run.status === 'completed');
            if (completed.length === 0) {
                this.updateExperimentStatus('No completed experiment runs found.', true);
                return;
            }
            const latest = completed[completed.length - 1];
            this.updateExperimentStatus(`Loaded run ${latest.run_id}.`);
            window.dispatchEvent(new CustomEvent('experiment-run-completed', { detail: { run_id: latest.run_id } }));
        } catch (error) {
            this.updateExperimentStatus(`Failed to load run: ${error.message}`, true);
        }
    }

    saveExperimentPreset() {
        try {
            const manifest = this.buildIntrinsicManifest();
            localStorage.setItem('intrinsic-dashboard-preset', JSON.stringify(manifest));
            this.updateExperimentStatus('Saved experiment preset locally.');
        } catch (error) {
            this.updateExperimentStatus(`Failed to save preset: ${error.message}`, true);
        }
    }

    loadExperimentPreset() {
        try {
            const raw = localStorage.getItem('intrinsic-dashboard-preset');
            if (!raw) {
                this.updateExperimentStatus('No saved preset found.', true);
                return;
            }
            const manifest = JSON.parse(raw);
            document.getElementById('exp-name').value = manifest.experiment_name || 'intrinsic-dashboard-run';
            document.getElementById('exp-steps').value = manifest.experiment_config?.num_steps || 400;
            document.getElementById('exp-snapshot-interval').value = manifest.experiment_config?.snapshot_interval || 25;
            document.getElementById('exp-selection-pressure').value = manifest.experiment_config?.policy?.selection_pressure || 'medium';
            document.getElementById('exp-enable-speciation').checked = Boolean(
                manifest.experiment_config?.speciation?.enabled
            );
            this.updateExperimentStatus('Loaded saved experiment preset.');
        } catch (error) {
            this.updateExperimentStatus(`Failed to load preset: ${error.message}`, true);
        }
    }
}

// Initialize sidebar when document is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sidebar = new Sidebar();
}); 
