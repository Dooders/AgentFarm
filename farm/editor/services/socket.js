const io = require('socket.io-client');

class SocketService {
    constructor() {
        this.socket = io('http://localhost:5000');
        this.setupListeners();
    }

    setupListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });

        this.socket.on('simulation_progress', (data) => {
            window.dispatchEvent(new CustomEvent('simulation-progress', { 
                detail: data 
            }));
        });

        this.socket.on('subscription_success', (data) => {
            console.log('Subscribed to simulation:', data.sim_id);
        });

        this.socket.on('subscription_error', (data) => {
            console.error('Subscription error:', data.message);
        });
    }

    subscribeToSimulation(simId) {
        this.socket.emit('subscribe_simulation', simId);
    }

    disconnect() {
        this.socket.disconnect();
    }
}

module.exports = new SocketService(); 