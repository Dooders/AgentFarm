/** @jest-environment jsdom */

// Minimal DOM bootstrap to load our explorer script
function injectHtml() {
    document.body.innerHTML = `
        <div id="app">
            <div id="sidebar"></div>
            <div id="main-content"></div>
        </div>
        <div id="config-explorer" style="display:none;"></div>
    `
}

// Mock schema service before loading Explorer
beforeEach(() => {
    injectHtml()
    window.configSchemaService = {
        fetchSchema: async () => ({
            version: 1,
            sections: {
                simulation: { title: 'Simulation', properties: { width: { type: 'integer' } } },
                visualization: { title: 'Visualization', properties: { zoom: { type: 'number' } } },
            },
        }),
    }
})

test('Explorer renders dynamic section buttons and selects first', async () => {
    await import('../components/config-explorer/Explorer.js')
    // Wait a tick for async init
    await new Promise((r) => setTimeout(r, 0))

    const explorer = document.getElementById('config-explorer')
    expect(explorer).toBeTruthy()
    // Show explorer
    window.showConfigExplorer()
    // Sections list should contain buttons
    const buttons = explorer.querySelectorAll('.section-item')
    expect(buttons.length).toBeGreaterThanOrEqual(2)
    // First section should be active
    const active = explorer.querySelector('.section-item.active')
    expect(active).toBeTruthy()
    // Details title should match first section
    const title = explorer.querySelector('#details-title').textContent
    expect(title).toBe('Simulation')
})

test('Explorer updates details when selecting a different section', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const vizBtn = Array.from(explorer.querySelectorAll('.section-item')).find(
        (b) => b.textContent === 'Visualization'
    )
    expect(vizBtn).toBeTruthy()
    vizBtn.click()
    const title = explorer.querySelector('#details-title').textContent
    expect(title).toBe('Visualization')
})

test('YAML preview placeholder is present', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const yamlHeader = explorer.querySelector('.yaml-header')
    const yamlCode = explorer.querySelector('.yaml-code')
    expect(yamlHeader).toBeTruthy()
    expect(yamlCode).toBeTruthy()
    expect(yamlCode.textContent).toContain('YAML preview will appear here')
})

