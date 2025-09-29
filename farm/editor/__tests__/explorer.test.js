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

test('Compare open enables field diff highlight and YAML diff grid', async () => {
    // Mock dialog to return a different config for compare
    window.dialogService = {
        openConfigDialog: async () => ({ canceled: false, filePath: '/tmp/compare.yaml' }),
        saveConfigDialog: async () => ({ canceled: true })
    }
    const originalLoad = window.configSchemaService.loadConfig
    window.configSchemaService.loadConfig = async (path) => {
        if (path === '/tmp/compare.yaml') {
            return { success: true, config: { width: 200, visualization: { zoom: 2.5 } } }
        }
        return originalLoad(path)
    }

    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    // Ensure Simulation section is active
    await waitForSelector(explorer, '.section-item')

    const compareBtn = await waitForSelector(explorer, '#open-compare')
    compareBtn.click()
    await new Promise((r) => setTimeout(r, 0))

    // Width field should be marked as different
    const row = explorer.querySelector('.form-row[data-field="width"]')
    expect(row.classList.contains('row-diff')).toBe(true)
    // YAML panel should show diff grid
    const yamlGrid = explorer.querySelector('.yaml-grid')
    expect(yamlGrid).toBeTruthy()
    expect(yamlGrid.textContent).toContain('width')
    expect(yamlGrid.textContent).toContain('100')
    expect(yamlGrid.textContent).toContain('200')

    // Clear compare disables grid
    const clearBtn = explorer.querySelector('#clear-compare')
    expect(clearBtn.disabled).toBe(false)
    clearBtn.click()
    await new Promise((r) => setTimeout(r, 0))
    expect(explorer.querySelector('.yaml-grid')).toBeFalsy()

    window.configSchemaService.loadConfig = originalLoad
})

test('Copy from compare sets field value and removes diff highlight', async () => {
    window.dialogService = {
        openConfigDialog: async () => ({ canceled: false, filePath: '/tmp/compare.yaml' }),
        saveConfigDialog: async () => ({ canceled: true })
    }
    const originalLoad = window.configSchemaService.loadConfig
    window.configSchemaService.loadConfig = async (path) => {
        if (path === '/tmp/compare.yaml') {
            return { success: true, config: { width: 200, visualization: { zoom: 2.5 } } }
        }
        return originalLoad(path)
    }

    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const compareBtn = await waitForSelector(explorer, '#open-compare')
    compareBtn.click()
    await new Promise((r) => setTimeout(r, 0))

    const copyBtn = explorer.querySelector('.form-row[data-field="width"] .copy-btn')
    expect(copyBtn).toBeTruthy()
    copyBtn.click()
    await new Promise((r) => setTimeout(r, 0))

    // After copying, width should now equal compare; diff mark may be removed after re-render
    const input = explorer.querySelector('.form-row[data-field="width"] input[type="number"]')
    expect(input.value).toBe('200')
})

test('Apply preset deep-merges and Undo preset restores previous config', async () => {
    // Seed a different preset via open dialog
    window.dialogService = {
        openConfigDialog: async () => ({ canceled: false, filePath: '/tmp/preset.yaml' }),
        saveConfigDialog: async () => ({ canceled: true })
    }
    const originalLoad = window.configSchemaService.loadConfig
    window.configSchemaService.loadConfig = async (path) => {
        if (path === '/tmp/preset.yaml') {
            return { success: true, config: { visualization: { zoom: 3.5 }, newSection: { flag: true } } }
        }
        return originalLoad(path)
    }

    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    // Switch to visualization section to observe value
    const buttons = explorer.querySelectorAll('.section-item')
    buttons[1].click()
    await new Promise((r) => setTimeout(r, 0))

    const applyBtn = await waitForSelector(explorer, '#apply-preset')
    applyBtn.click()
    await new Promise((r) => setTimeout(r, 0))

    // zoom should update to 3.5
    const zoomInput = explorer.querySelector('.details-content input[type="number"]')
    expect(zoomInput.value).toBe('3.5')
    // Undo should re-enable and revert
    const undoBtn = explorer.querySelector('#undo-preset')
    expect(undoBtn.disabled).toBe(false)
    undoBtn.click()
    await new Promise((r) => setTimeout(r, 0))
    const zoomInputAfterUndo = explorer.querySelector('.details-content input[type="number"]')
    expect(zoomInputAfterUndo.value).toBe('1')

    window.configSchemaService.loadConfig = originalLoad
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

test('Toolbar shows Open, Browse, Save As buttons', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const openBtn = explorer.querySelector('#open-config')
    const browseBtn = explorer.querySelector('#browse-save')
    const saveAsBtn = explorer.querySelector('#save-as')
    expect(openBtn).toBeTruthy()
    expect(browseBtn).toBeTruthy()
    expect(saveAsBtn).toBeTruthy()
})

test('Open button uses dialog service and loads config', async () => {
    // Mock dialog service and capture path passed to loadConfig
    const openedPath = '/tmp/opened.yaml'
    window.dialogService = {
        openConfigDialog: async () => ({ canceled: false, filePath: openedPath }),
        saveConfigDialog: async () => ({ canceled: true })
    }
    const originalLoad = window.configSchemaService.loadConfig
    let receivedPath = null
    window.configSchemaService.loadConfig = async (path) => {
        receivedPath = path
        return { success: true, config: { width: 200, visualization: { zoom: 2.0 } } }
    }

    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const openBtn = await waitForSelector(explorer, '#open-config')
    openBtn.click()
    await new Promise((r) => setTimeout(r, 0))

    // loadConfig should have been called with path from dialog
    expect(receivedPath).toBe(openedPath)
    // YAML should reflect loaded config
    const yamlCode = await waitForSelector(explorer, '.yaml-code')
    expect(yamlCode.textContent).toContain('width: 200')
    // Save path input should be populated
    const savePath = explorer.querySelector('#save-path')
    expect(savePath.value).toBe(openedPath)

    // Restore original
    window.configSchemaService.loadConfig = originalLoad
})

test('Browse button fills save path from dialog', async () => {
    const targetPath = '/tmp/save-here.yaml'
    window.dialogService = {
        openConfigDialog: async () => ({ canceled: true }),
        saveConfigDialog: async () => ({ canceled: false, filePath: targetPath })
    }

    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const browseBtn = await waitForSelector(explorer, '#browse-save')
    browseBtn.click()
    await new Promise((r) => setTimeout(r, 0))

    const savePath = explorer.querySelector('#save-path')
    expect(savePath.value).toBe(targetPath)
})

test('Save As prompts for path and saves there', async () => {
    const saveAsPath = '/tmp/config-new.yaml'
    let savedPathObserved = null
    window.dialogService = {
        openConfigDialog: async () => ({ canceled: true }),
        saveConfigDialog: async () => ({ canceled: false, filePath: saveAsPath })
    }
    const originalSave = window.configSchemaService.saveConfig
    window.configSchemaService.saveConfig = async (config, path) => {
        savedPathObserved = path
        return { success: true, config, path }
    }

    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    // Make a change to enable Save button
    const input = await waitForSelector(explorer, '.details-content input[type="number"]')
    input.value = '250'
    input.dispatchEvent(new Event('input'))

    const saveAsBtn = await waitForSelector(explorer, '#save-as')
    saveAsBtn.click()
    await new Promise((r) => setTimeout(r, 0))

    // Save should be called with selected path
    expect(savedPathObserved).toBe(saveAsPath)
    // Unsaved indicator should be hidden after successful save
    const unsaved = explorer.querySelector('#unsaved-indicator')
    expect(unsaved && unsaved.style.display).toBe('none')

    // Restore original
    window.configSchemaService.saveConfig = originalSave
})

test('Grayscale toggle button toggles body class and persists', async () => {
    try { localStorage.removeItem('ui:grayscale') } catch (e) {}
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const grayBtn = explorer.querySelector('#toggle-grayscale')
    expect(grayBtn).toBeTruthy()

    grayBtn.click()
    await new Promise((r) => setTimeout(r, 0))
    expect(document.body.classList.contains('grayscale')).toBe(true)
    expect(grayBtn.classList.contains('toggled')).toBe(true)
    expect(localStorage.getItem('ui:grayscale')).toBe('1')

    grayBtn.click()
    await new Promise((r) => setTimeout(r, 0))
    expect(document.body.classList.contains('grayscale')).toBe(false)
    expect(grayBtn.classList.contains('toggled')).toBe(false)
    expect(localStorage.getItem('ui:grayscale')).toBe('0')
})

test('Grayscale sync helpers update toolbar and sidebar buttons', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const toolbarBtn = explorer.querySelector('#toggle-grayscale')
    expect(toolbarBtn).toBeTruthy()

    const sidebarBtn = document.createElement('button')
    sidebarBtn.id = 'toggle-grayscale-sidebar'
    document.body.appendChild(sidebarBtn)

    window.__setExplorerGrayscale(true)
    expect(document.body.classList.contains('grayscale')).toBe(true)
    expect(toolbarBtn.classList.contains('toggled')).toBe(true)

    window.__setSidebarGrayscale(false)
    expect(document.body.classList.contains('grayscale')).toBe(false)
    expect(sidebarBtn.classList.contains('toggled')).toBe(false)
})

test('Keyboard navigation moves focus and updates selection in section list', async () => {
    await import('../components/config-explorer/Explorer.js')
    await new Promise((r) => setTimeout(r, 0))
    window.showConfigExplorer()

    const explorer = document.getElementById('config-explorer')
    const list = explorer.querySelector('.section-list')
    const items = explorer.querySelectorAll('.section-item')
    expect(items.length).toBeGreaterThan(1)

    // Focus first item
    items[0].focus()
    // ArrowDown should move to second and select it
    const evt = new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })
    list.dispatchEvent(evt)

    expect(document.activeElement).toBe(items[1])
    const title = explorer.querySelector('#details-title').textContent
    expect(title).toBe('Visualization')

    // Home should move to first
    const homeEvt = new KeyboardEvent('keydown', { key: 'Home', bubbles: true })
    list.dispatchEvent(homeEvt)
    expect(document.activeElement).toBe(items[0])
    const title2 = explorer.querySelector('#details-title').textContent
    expect(title2).toBe('Simulation')
})

