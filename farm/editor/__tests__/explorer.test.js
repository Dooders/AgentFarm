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

async function waitForSelector(root, selector, timeoutMs = 1000) {
    const start = Date.now()
    return new Promise((resolve, reject) => {
        (function check() {
            const el = root.querySelector(selector)
            if (el) return resolve(el)
            if (Date.now() - start > timeoutMs) return reject(new Error(`Timeout waiting for ${selector}`))
            setTimeout(check, 10)
        })()
    })
}

// Mock schema service before loading Explorer
beforeEach(() => {
    jest.resetModules()
    injectHtml()
    window.configSchemaService = {
        fetchSchema: async () => ({
            version: 1,
            sections: {
                simulation: { title: 'Simulation', properties: { width: { type: 'integer' } } },
                visualization: { title: 'Visualization', properties: { zoom: { type: 'number' } } },
            },
        }),
        loadConfig: async () => ({ success: true, config: { width: 100, visualization: { zoom: 1.0 } } }),
        validateConfig: async (config) => ({ success: true, message: 'Configuration is valid', config }),
        saveConfig: async (config, path) => ({ success: true, message: 'Configuration saved', config, path: path || '/tmp/config.yaml' })
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
    await waitForSelector(explorer, '.section-item')
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
    await waitForSelector(explorer, '.section-item')
    const buttons = explorer.querySelectorAll('.section-item')
    expect(buttons.length).toBeGreaterThan(1)
    buttons[1].click()
    const title = explorer.querySelector('#details-title').textContent
    expect(title).toBe('Visualization')
})

test('YAML preview renders current config', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const yamlHeader = await waitForSelector(explorer, '.yaml-header')
    const yamlCode = await waitForSelector(explorer, '.yaml-code')
    expect(yamlHeader).toBeTruthy()
    expect(yamlCode).toBeTruthy()
    expect(yamlCode.textContent).toContain('width: 100')
})

test('Editing a field marks unsaved and updates YAML', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    // Ensure Simulation section is active
    const input = await waitForSelector(explorer, '.details-content input[type="number"]')
    expect(input).toBeTruthy()
    input.value = '120'
    input.dispatchEvent(new Event('input'))

    // Unsaved indicator should appear
    const unsaved = await waitForSelector(explorer, '#unsaved-indicator')
    expect(unsaved && unsaved.style.display).toBe('')

    const yamlCode = await waitForSelector(explorer, '.yaml-code')
    expect(yamlCode.textContent).toContain('width: 120')
})

test('Save button calls save service and clears unsaved', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const input = await waitForSelector(explorer, '.details-content input[type="number"]')
    input.value = '150'
    input.dispatchEvent(new Event('input'))

    const saveBtn = await waitForSelector(explorer, '#save-config')
    saveBtn.click()
    // wait for async save
    await new Promise((r) => setTimeout(r, 0))

    const unsaved = explorer.querySelector('#unsaved-indicator')
    expect(unsaved && unsaved.style.display).toBe('none')

    const header = explorer.querySelector('.yaml-header')
    expect(header.innerHTML).toContain('badge valid')
})

test('Client validation shows message and row-error for bounds', async () => {
    // Override schema with min/max for width
    window.configSchemaService.fetchSchema = async () => ({
        version: 1,
        sections: {
            simulation: { title: 'Simulation', properties: { width: { type: 'integer', minimum: 10, maximum: 200 } } },
        },
    })
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const input = await waitForSelector(explorer, '.details-content input[type="number"]')
    input.value = '5'
    input.dispatchEvent(new Event('input'))

    // Row should have error class and message
    const row = explorer.querySelector('.form-row[data-field="width"]')
    expect(row.classList.contains('row-error')).toBe(true)
    const msg = row.querySelector('.validation-msg').textContent
    expect(msg).toMatch(/Must be ≥ 10/)
})

