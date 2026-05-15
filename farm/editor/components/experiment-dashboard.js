class ExperimentDashboard {
    constructor() {
        this.statsRoot = document.getElementById('stats');
        this.currentRunId = null;
        this.currentViews = [];
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
                    <select id="dashboard-view-select"></select>
                    <button id="dashboard-load-view">Load View</button>
                </div>
                <div id="dashboard-body">Run an experiment from the builder to load views.</div>
            </div>
        `;

        document.getElementById('dashboard-load-view').addEventListener('click', () => {
            this.loadSelectedView();
        });

        window.addEventListener('experiment-run-completed', (event) => {
            this.loadRun(event.detail.run_id);
        });
    }

    async loadRun(runId) {
        this.currentRunId = runId;
        document.getElementById('dashboard-run-id').textContent = `Run: ${runId}`;
        const body = document.getElementById('dashboard-body');
        body.textContent = 'Loading views...';
        try {
            const response = await fetch(`http://localhost:5000/api/experiments/${runId}/views`);
            const payload = await response.json();
            this.currentViews = payload?.data || [];
            this.renderViewOptions();
            await this.loadSelectedView();
        } catch (error) {
            body.textContent = `Failed to load views: ${error.message}`;
        }
    }

    renderViewOptions() {
        const select = document.getElementById('dashboard-view-select');
        select.innerHTML = '';
        this.currentViews.forEach((view) => {
            const option = document.createElement('option');
            option.value = view.view_id;
            option.textContent = view.title;
            select.appendChild(option);
        });
    }

    async loadSelectedView() {
        if (!this.currentRunId) {
            return;
        }
        const select = document.getElementById('dashboard-view-select');
        const viewId = select.value;
        if (!viewId) {
            return;
        }
        const body = document.getElementById('dashboard-body');
        body.textContent = 'Loading view data...';
        try {
            const response = await fetch(
                `http://localhost:5000/api/experiments/${this.currentRunId}/views/${viewId}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filters: {} })
                }
            );
            const payload = await response.json();
            this.renderViewData(payload?.data || {});
        } catch (error) {
            body.textContent = `Failed to load view data: ${error.message}`;
        }
    }

    renderViewData(data) {
        const body = document.getElementById('dashboard-body');
        if (data.view_type === 'summary_cards') {
            const cards = (data.cards || [])
                .map((card) => `<div class="dashboard-card"><strong>${card.label}</strong><span>${card.value}</span></div>`)
                .join('');
            body.innerHTML = `<div class="dashboard-card-grid">${cards}</div>`;
            return;
        }

        if (data.view_type === 'timeseries') {
            const rows = (data.series || [])
                .map((series) => `<tr><td>${series.label}</td><td>${(series.values || []).slice(-1)[0] ?? 'n/a'}</td><td>${(series.values || []).length}</td></tr>`)
                .join('');
            body.innerHTML = `
                <h4>${data.title || 'Time series'}</h4>
                <table class="dashboard-table">
                    <thead><tr><th>Series</th><th>Latest</th><th>Samples</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
            return;
        }

        if (data.view_type === 'distribution_over_time') {
            const rows = (data.snapshots || [])
                .map((snapshot) => {
                    const genes = Object.keys(snapshot.by_gene || {}).length;
                    return `<tr><td>${snapshot.step}</td><td>${genes}</td></tr>`;
                })
                .join('');
            body.innerHTML = `
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
            body.innerHTML = `
                <h4>${data.title || 'Lineage and clusters'}</h4>
                <div class="dashboard-card-grid">
                    <div class="dashboard-card"><strong>Latest Speciation Index</strong><span>${latestIndex}</span></div>
                    <div class="dashboard-card"><strong>Cluster Records</strong><span>${clusterCount}</span></div>
                </div>
            `;
            return;
        }

        body.textContent = 'No renderer available for this view.';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.experimentDashboard = new ExperimentDashboard();
});
