const API_BASE_URL = 'http://192.168.1.182:8000';

async function loadTree(colorBy = 'offspring', maxGenerations = null) {
    const params = new URLSearchParams();
    if (colorBy) params.append('color_by', colorBy);
    if (maxGenerations) params.append('max_generations', maxGenerations);
    
    const iframe = document.getElementById('tree-frame');
    iframe.src = `${API_BASE_URL}/tree?${params.toString()}`;
}

async function refreshTree() {
    const colorBy = document.getElementById('color-by').value;
    const maxGenerations = document.getElementById('max-generations').value;
    
    try {
        // Force cache refresh
        await fetch(`${API_BASE_URL}/tree/refresh`);
        // Load the tree with new parameters
        await loadTree(colorBy, maxGenerations || null);
    } catch (error) {
        console.error('Failed to refresh tree:', error);
    }
}

// Initialize
window.addEventListener('load', () => {
    loadTree();
}); 