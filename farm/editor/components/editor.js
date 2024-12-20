const axios = require('axios');
const Chart = require('chart.js');
const socketService = require('../services/socket');

class Editor {
    constructor() {
        this.visualization = document.getElementById('visualization');
        this.controls = document.getElementById('controls');
        this.stats = document.getElementById('stats');
        this.currentSimId = null;
        this.isPlaying = false;
        this.currentStep = 0;
        this.playbackSpeed = 5;
        this.charts = {};
        
        this.init();
    }

    init() {
        this.setupVisualization();
        this.setupControls();
        this.setupStats();
        this.setupCharts();
        this.attachEventListeners();
    }

    setupVisualization() {
        // Setup main canvas
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'sim-canvas';
        this.visualization.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');

        // Add loading overlay
        this.loadingOverlay = document.createElement('div');
        this.loadingOverlay.className = 'loading-overlay';
        this.loadingOverlay.innerHTML = `
            <div class="loading-content">
                <div class="spinner"></div>
                <div class="progress-text">Starting simulation...</div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            </div>
        `;
        this.visualization.appendChild(this.loadingOverlay);

        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    setupCharts() {
        const chartsContainer = document.createElement('div');
        chartsContainer.className = 'charts-container';
        this.visualization.appendChild(chartsContainer);

        // Population chart
        const popCanvas = document.createElement('canvas');
        chartsContainer.appendChild(popCanvas);
        this.charts.population = new Chart(popCanvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'System Agents',
                        data: [],
                        borderColor: 'blue',
                        fill: false
                    },
                    {
                        label: 'Independent Agents',
                        data: [],
                        borderColor: 'red',
                        fill: false
                    },
                    {
                        label: 'Control Agents',
                        data: [],
                        borderColor: 'yellow',
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: 'Population Dynamics'
                }
            }
        });

        // Resources chart
        const resCanvas = document.createElement('canvas');
        chartsContainer.appendChild(resCanvas);
        this.charts.resources = new Chart(resCanvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Resources',
                    data: [],
                    borderColor: 'green',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: 'Resource Distribution'
                }
            }
        });
    }

    setupControls() {
        this.controls.innerHTML = `
            <div class="control-panel">
                <button id="play-pause" class="control-btn">▶</button>
                <button id="step-back" class="control-btn">◀</button>
                <button id="step-forward" class="control-btn">▶</button>
                <div class="speed-control">
                    <input type="range" id="speed" min="1" max="10" value="5">
                    <span>Speed: <span id="speed-value">5</span></span>
                </div>
                <button id="restart" class="control-btn">⟲</button>
                <button id="analyze" class="control-btn"></button>
                <button id="export" class="control-btn">💾</button>
            </div>
        `;

        this.attachControlListeners();
    }

    attachControlListeners() {
        document.getElementById('play-pause').onclick = () => this.togglePlayback();
        document.getElementById('step-back').onclick = () => this.step(-1);
        document.getElementById('step-forward').onclick = () => this.step(1);
        document.getElementById('restart').onclick = () => this.restart();
        document.getElementById('analyze').onclick = () => this.showAnalysis();
        document.getElementById('export').onclick = () => this.exportData();

        document.getElementById('speed').oninput = (e) => {
            this.playbackSpeed = parseInt(e.target.value);
            document.getElementById('speed-value').textContent = this.playbackSpeed;
        };
    }

    attachEventListeners() {
        window.addEventListener('simulation-created', (e) => {
            this.handleSimulationCreated(e.detail);
        });

        window.addEventListener('simulation-progress', (e) => {
            this.handleSimulationProgress(e.detail);
        });

        window.addEventListener('simulation-updated', (e) => {
            this.handleSimulationUpdated(e.detail);
        });
    }

    async handleSimulationCreated(data) {
        this.currentSimId = data.sim_id;
        this.currentStep = 0;
        socketService.subscribeToSimulation(this.currentSimId);
        this.updateVisualization(data.data);
        this.hideLoading();
    }

    handleSimulationProgress(data) {
        if (data.sim_id === this.currentSimId) {
            const progressFill = this.loadingOverlay.querySelector('.progress-fill');
            progressFill.style.width = `${data.percentage}%`;
            
            const progressText = this.loadingOverlay.querySelector('.progress-text');
            progressText.textContent = `Processing step ${data.step} of ${data.total_steps}...`;
        }
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    async handleSimulationUpdated(data) {
        this.updateVisualization(data);
        this.updateStats(data);
        this.updateCharts(data);
    }

    updateVisualization(data) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw environment grid
        this.drawGrid();

        // Draw resources
        if (data.resources) {
            data.resources.forEach(resource => {
                this.drawResource(resource);
            });
        }

        // Draw agents
        if (data.agents) {
            data.agents.forEach(agent => {
                this.drawAgent(agent);
            });
        }

        // Draw interactions if any
        if (data.interactions) {
            data.interactions.forEach(interaction => {
                this.drawInteraction(interaction);
            });
        }
    }

    drawGrid() {
        const gridSize = 20;
        this.ctx.strokeStyle = '#2c3e50';
        this.ctx.lineWidth = 0.5;
        this.ctx.globalAlpha = 0.1;

        for (let x = 0; x <= this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }

        for (let y = 0; y <= this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }

        this.ctx.globalAlpha = 1;
    }

    drawAgent(agent) {
        const x = this.scaleX(agent.x);
        const y = this.scaleY(agent.y);
        const radius = 5;

        // Draw agent body
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = this.getAgentColor(agent.type);
        this.ctx.fill();

        // Draw health bar
        if (agent.health !== undefined) {
            this.drawHealthBar(x, y - radius - 5, agent.health);
        }

        // Draw resource indicator
        if (agent.resources !== undefined) {
            this.drawResourceIndicator(x, y + radius + 5, agent.resources);
        }

        // Draw selection highlight if agent is selected
        if (agent.selected) {
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }
    }

    drawHealthBar(x, y, health) {
        const width = 20;
        const height = 3;
        
        // Background
        this.ctx.fillStyle = '#e74c3c';
        this.ctx.fillRect(x - width/2, y, width, height);
        
        // Health level
        this.ctx.fillStyle = '#2ecc71';
        this.ctx.fillRect(x - width/2, y, width * (health/100), height);
    }

    drawResourceIndicator(x, y, resources) {
        const radius = 2;
        const color = this.getResourceColor(resources);
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
    }

    drawResource(resource) {
        const x = this.scaleX(resource.x);
        const y = this.scaleY(resource.y);
        const size = Math.min(10, resource.amount / 10);

        // Draw glow effect
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, size * 2);
        gradient.addColorStop(0, 'rgba(46, 204, 113, 0.3)');
        gradient.addColorStop(1, 'rgba(46, 204, 113, 0)');
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, size * 2, 0, Math.PI * 2);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();

        // Draw resource
        this.ctx.beginPath();
        this.ctx.arc(x, y, size, 0, Math.PI * 2);
        this.ctx.fillStyle = '#2ecc71';
        this.ctx.fill();
    }

    drawInteraction(interaction) {
        const startX = this.scaleX(interaction.source.x);
        const startY = this.scaleY(interaction.source.y);
        const endX = this.scaleX(interaction.target.x);
        const endY = this.scaleY(interaction.target.y);

        // Draw interaction line
        this.ctx.beginPath();
        this.ctx.moveTo(startX, startY);
        this.ctx.lineTo(endX, endY);
        this.ctx.strokeStyle = this.getInteractionColor(interaction.type);
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw interaction effect
        this.drawInteractionEffect(interaction);
    }

    drawInteractionEffect(interaction) {
        const centerX = (this.scaleX(interaction.source.x) + this.scaleX(interaction.target.x)) / 2;
        const centerY = (this.scaleY(interaction.source.y) + this.scaleY(interaction.target.y)) / 2;

        switch (interaction.type) {
            case 'attack':
                this.drawExplosion(centerX, centerY);
                break;
            case 'share':
                this.drawTransfer(centerX, centerY);
                break;
            case 'reproduce':
                this.drawSpawn(centerX, centerY);
                break;
        }
    }

    drawExplosion(x, y) {
        const radius = 10;
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, 'rgba(231, 76, 60, 0.8)');  // Red center
        gradient.addColorStop(0.7, 'rgba(241, 196, 15, 0.5)');  // Yellow mid
        gradient.addColorStop(1, 'rgba(241, 196, 15, 0)');  // Transparent edge

        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
    }

    drawTransfer(x, y) {
        const size = 8;
        const particles = 6;
        const angleStep = (Math.PI * 2) / particles;

        for (let i = 0; i < particles; i++) {
            const angle = i * angleStep;
            const particleX = x + Math.cos(angle) * size;
            const particleY = y + Math.sin(angle) * size;

            this.ctx.beginPath();
            this.ctx.arc(particleX, particleY, 2, 0, Math.PI * 2);
            this.ctx.fillStyle = '#3498db';  // Blue
            this.ctx.fill();
        }
    }

    drawSpawn(x, y) {
        const size = 12;
        this.ctx.beginPath();
        this.ctx.arc(x, y, size, 0, Math.PI * 2);
        this.ctx.strokeStyle = '#2ecc71';  // Green
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([2, 2]);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    // Playback Controls
    async togglePlayback() {
        this.isPlaying = !this.isPlaying;
        const button = document.getElementById('play-pause');
        button.textContent = this.isPlaying ? '⏸' : '▶';

        if (this.isPlaying) {
            this.play();
        }
    }

    async play() {
        if (!this.isPlaying) return;

        try {
            const response = await axios.get(
                `http://localhost:5000/api/simulation/${this.currentSimId}/step/${this.currentStep + 1}`
            );

            if (response.data.status === 'success') {
                this.currentStep++;
                this.handleSimulationUpdated(response.data.data);
                
                // Schedule next step based on playback speed
                setTimeout(() => this.play(), 1000 / this.playbackSpeed);
            } else {
                // End of simulation reached
                this.isPlaying = false;
                document.getElementById('play-pause').textContent = '▶';
            }
        } catch (error) {
            console.error('Error during playback:', error);
            this.isPlaying = false;
            document.getElementById('play-pause').textContent = '▶';
        }
    }

    async step(direction) {
        const targetStep = this.currentStep + direction;
        if (targetStep < 0) return;

        try {
            const response = await axios.get(
                `http://localhost:5000/api/simulation/${this.currentSimId}/step/${targetStep}`
            );

            if (response.data.status === 'success') {
                this.currentStep = targetStep;
                this.handleSimulationUpdated(response.data.data);
            }
        } catch (error) {
            console.error('Error stepping simulation:', error);
        }
    }

    async restart() {
        try {
            const response = await axios.get(
                `http://localhost:5000/api/simulation/${this.currentSimId}/step/0`
            );

            if (response.data.status === 'success') {
                this.currentStep = 0;
                this.handleSimulationUpdated(response.data.data);
                this.isPlaying = false;
                document.getElementById('play-pause').textContent = '▶';
            }
        } catch (error) {
            console.error('Error restarting simulation:', error);
        }
    }

    // Analysis and Charts
    async showAnalysis() {
        try {
            const response = await axios.get(
                `http://localhost:5000/api/simulation/${this.currentSimId}/analysis`
            );

            if (response.data.status === 'success') {
                this.showAnalysisModal(response.data.data);
            }
        } catch (error) {
            console.error('Error getting analysis:', error);
        }
    }

    showAnalysisModal(data) {
        const modal = document.createElement('div');
        modal.className = 'analysis-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close">&times;</span>
                <h2>Simulation Analysis</h2>
                <div class="analysis-tabs">
                    <button class="tab-btn active" data-tab="population">Population</button>
                    <button class="tab-btn" data-tab="resources">Resources</button>
                    <button class="tab-btn" data-tab="behavior">Behavior</button>
                </div>
                <div class="tab-content">
                    ${this.renderAnalysisTabs(data)}
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        this.attachAnalysisEventListeners(modal);
    }

    renderAnalysisTabs(data) {
        return `
            <div id="population" class="tab-panel active">
                ${this.renderPopulationAnalysis(data.population)}
            </div>
            <div id="resources" class="tab-panel">
                ${this.renderResourceAnalysis(data.resources)}
            </div>
            <div id="behavior" class="tab-panel">
                ${this.renderBehaviorAnalysis(data.behavior)}
            </div>
        `;
    }

    // Helper methods
    scaleX(x) {
        return (x / 100) * this.canvas.width;
    }

    scaleY(y) {
        return (y / 100) * this.canvas.height;
    }

    getAgentColor(type) {
        const colors = {
            'system': '#3498db',     // Blue
            'independent': '#e74c3c', // Red
            'control': '#f1c40f'      // Yellow
        };
        return colors[type] || '#95a5a6';  // Gray default
    }

    getResourceColor(amount) {
        const maxResources = 100;
        const intensity = Math.min(amount / maxResources, 1);
        return `rgba(46, 204, 113, ${intensity})`;  // Green with varying opacity
    }

    getInteractionColor(type) {
        const colors = {
            'attack': '#e74c3c',    // Red
            'share': '#3498db',     // Blue
            'reproduce': '#2ecc71'  // Green
        };
        return colors[type] || '#95a5a6';  // Gray default
    }
}

// Initialize editor when document is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.editor = new Editor();
}); 