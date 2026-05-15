class ExperimentDashboard {
    constructor() {
        this.statsRoot = document.getElementById('stats');
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
            const cards = (data.cards || [])
                .map((card) => `<div class="dashboard-card"><strong>${card.label}</strong><span>${card.value}</span></div>`)
                .join('');
            if (!cards) {
                this.body.textContent = 'No summary cards are available for this run.';
                return;
            }
            this.body.innerHTML = `<div class="dashboard-card-grid">${cards}</div>`;
            return;
        }

        if (data.view_type === 'timeseries') {
            if (!data.series?.length) {
                this.body.textContent = 'No time series data is available for this run.';
                return;
            }
            const rows = (data.series || [])
                .map((series) => `<tr><td>${series.label}</td><td>${(series.values || []).slice(-1)[0] ?? 'n/a'}</td><td>${(series.values || []).length}</td></tr>`)
                .join('');
            this.body.innerHTML = `
                <h4>${data.title || 'Time series'}</h4>
                <table class="dashboard-table">
                    <thead><tr><th>Series</th><th>Latest</th><th>Samples</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
            return;
        }

        if (data.view_type === 'distribution_over_time') {
            if (!data.snapshots?.length) {
                this.body.textContent = 'No distribution snapshots are available for this run.';
                return;
            }
            const rows = (data.snapshots || [])
                .map((snapshot) => {
                    const genes = Object.keys(snapshot.by_gene || {}).length;
                    return `<tr><td>${snapshot.step}</td><td>${genes}</td></tr>`;
                })
                .join('');
            this.body.innerHTML = `
                <h4>${data.title || 'Distribution over time'}</h4>
                <table class="dashboard-table">
                    <thead><tr><th>Step</th><th>Genes tracked</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
            return;
        }

        if (data.view_type === 'lineage_or_clusters') {
            const clusterCount = (data.clusters || []).length;
            const latestIndex = data?.timeseries?.series?.[0]?.values?.slice(-1)[0] ?? 'n/a';
            this.body.innerHTML = `
                <h4>${data.title || 'Lineage and clusters'}</h4>
                <div class="dashboard-card-grid">
                    <div class="dashboard-card"><strong>Latest Speciation Index</strong><span>${latestIndex}</span></div>
                    <div class="dashboard-card"><strong>Cluster Records</strong><span>${clusterCount}</span></div>
                </div>
            `;
            return;
        }

        this.body.textContent = 'No renderer available for this view.';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.experimentDashboard = new ExperimentDashboard();
});
