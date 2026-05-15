class ExperimentDashboard {
    constructor() {
        this.statsRoot = document.getElementById('experiment-dashboard');
        if (!this.statsRoot) {
            return;
        }
        this.currentRunId = null;
        this.currentViews = [];
        this.isLoadingView = false;
        this.init();
    }

    init() {
        this.statsRoot.innerHTML = `
            <div class="dashboard-shell">
                <div class="dashboard-header">
                    <h3>Experiment Dashboard</h3>
                    <div id="dashboard-run-id">No run selected</div>
                </div>
                <div class="dashboard-controls">
                    <label for="dashboard-view-select">View:</label>
                    <select id="dashboard-view-select" disabled></select>
                    <button id="dashboard-load-view" disabled>Load View</button>
                </div>
                <div id="dashboard-body">Run an experiment from the builder to load views.</div>
            </div>
        `;

        this.viewSelect = document.getElementById('dashboard-view-select');
        this.loadButton = document.getElementById('dashboard-load-view');
        this.runLabel = document.getElementById('dashboard-run-id');
        this.body = document.getElementById('dashboard-body');

        this.loadButton.addEventListener('click', () => {
            this.loadSelectedView();
        });

        window.addEventListener('experiment-run-completed', (event) => {
            this.loadRun(event.detail.run_id);
        });
    }

    async loadRun(runId) {
        this.currentRunId = runId;
        this.runLabel.textContent = `Run: ${runId}`;
        this.body.textContent = 'Loading views...';
        this.viewSelect.disabled = true;
        this.loadButton.disabled = true;
        try {
            const response = await fetch(`http://localhost:5000/api/experiments/${runId}/views`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            this.currentViews = payload?.data || [];
            this.renderViewOptions();
            if (this.currentViews.length === 0) {
                this.body.textContent = 'No dashboard views are available for this run.';
                return;
            }
            await this.loadSelectedView();
        } catch (error) {
            this.body.textContent = `Failed to load views: ${error.message}`;
        }
    }

    renderViewOptions() {
        const previousSelection = this.viewSelect.value;
        this.viewSelect.innerHTML = '';
        this.currentViews.forEach((view) => {
            const option = document.createElement('option');
            option.value = view.view_id;
            option.textContent = view.title;
            this.viewSelect.appendChild(option);
        });
        if (previousSelection && this.currentViews.some((view) => view.view_id === previousSelection)) {
            this.viewSelect.value = previousSelection;
        }
        const hasViews = this.currentViews.length > 0;
        this.viewSelect.disabled = !hasViews;
        this.loadButton.disabled = !hasViews;
    }

    async loadSelectedView() {
        if (!this.currentRunId || this.isLoadingView) {
            return;
        }
        const viewId = this.viewSelect.value;
        if (!viewId) {
            this.body.textContent = 'Select a dashboard view to load.';
            return;
        }
        this.isLoadingView = true;
        this.body.textContent = 'Loading view data...';
        this.loadButton.disabled = true;
        try {
            const response = await fetch(
                `http://localhost:5000/api/experiments/${this.currentRunId}/views/${viewId}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filters: {} })
                }
            );
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            this.renderViewData(payload?.data || {});
        } catch (error) {
            this.body.textContent = `Failed to load view data: ${error.message}`;
        } finally {
            this.isLoadingView = false;
            this.loadButton.disabled = this.currentViews.length === 0;
        }
    }

    renderViewData(data) {
        if (data.view_type === 'summary_cards') {
            const cards = data.cards || [];
            if (!cards.length) {
                this.body.textContent = 'No summary cards are available for this run.';
                return;
            }
            const grid = document.createElement('div');
            grid.className = 'dashboard-card-grid';
            cards.forEach((card) => {
                const cardElement = document.createElement('div');
                cardElement.className = 'dashboard-card';
                const label = document.createElement('strong');
                label.textContent = String(card.label ?? '');
                const value = document.createElement('span');
                value.textContent = String(card.value ?? '');
                cardElement.appendChild(label);
                cardElement.appendChild(value);
                grid.appendChild(cardElement);
            });
            this.body.replaceChildren(grid);
            return;
        }

        if (data.view_type === 'timeseries') {
            if (!data.series?.length) {
                this.body.textContent = 'No time series data is available for this run.';
                return;
            }
            const title = document.createElement('h4');
            title.textContent = data.title || 'Time series';
            const table = document.createElement('table');
            table.className = 'dashboard-table';
            table.innerHTML = '<thead><tr><th>Series</th><th>Latest</th><th>Samples</th></tr></thead>';
            const body = document.createElement('tbody');
            (data.series || []).forEach((series) => {
                const row = document.createElement('tr');
                const label = document.createElement('td');
                label.textContent = String(series.label ?? '');
                const latest = document.createElement('td');
                latest.textContent = String((series.values || []).slice(-1)[0] ?? 'n/a');
                const samples = document.createElement('td');
                samples.textContent = String((series.values || []).length);
                row.appendChild(label);
                row.appendChild(latest);
                row.appendChild(samples);
                body.appendChild(row);
            });
            table.appendChild(body);
            this.body.replaceChildren(title, table);
            return;
        }

        if (data.view_type === 'distribution_over_time') {
            if (!data.snapshots?.length) {
                this.body.textContent = 'No distribution snapshots are available for this run.';
                return;
            }
            const title = document.createElement('h4');
            title.textContent = data.title || 'Distribution over time';
            const table = document.createElement('table');
            table.className = 'dashboard-table';
            table.innerHTML = '<thead><tr><th>Step</th><th>Genes tracked</th></tr></thead>';
            const body = document.createElement('tbody');
            (data.snapshots || []).forEach((snapshot) => {
                const row = document.createElement('tr');
                const step = document.createElement('td');
                step.textContent = String(snapshot.step ?? '');
                const genes = document.createElement('td');
                genes.textContent = String(Object.keys(snapshot.by_gene || {}).length);
                row.appendChild(step);
                row.appendChild(genes);
                body.appendChild(row);
            });
            table.appendChild(body);
            this.body.replaceChildren(title, table);
            return;
        }

        if (data.view_type === 'lineage_or_clusters') {
            const clusterCount = (data.clusters || []).length;
            const latestIndex = data?.timeseries?.series?.[0]?.values?.slice(-1)[0] ?? 'n/a';
            const title = document.createElement('h4');
            title.textContent = data.title || 'Lineage and clusters';
            const grid = document.createElement('div');
            grid.className = 'dashboard-card-grid';
            const indexCard = document.createElement('div');
            indexCard.className = 'dashboard-card';
            const indexLabel = document.createElement('strong');
            indexLabel.textContent = 'Latest Speciation Index';
            const indexValue = document.createElement('span');
            indexValue.textContent = String(latestIndex);
            indexCard.appendChild(indexLabel);
            indexCard.appendChild(indexValue);
            const clusterCard = document.createElement('div');
            clusterCard.className = 'dashboard-card';
            const clusterLabel = document.createElement('strong');
            clusterLabel.textContent = 'Cluster Records';
            const clusterValue = document.createElement('span');
            clusterValue.textContent = String(clusterCount);
            clusterCard.appendChild(clusterLabel);
            clusterCard.appendChild(clusterValue);
            grid.appendChild(indexCard);
            grid.appendChild(clusterCard);
            this.body.replaceChildren(title, grid);
            return;
        }

        this.body.textContent = 'No renderer available for this view.';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.experimentDashboard = new ExperimentDashboard();
});
