/** @jest-environment jsdom */

function injectHtml() {
    document.body.innerHTML = '<div id="sidebar"></div>'
}

function mockJsonResponse(payload, ok = true) {
    return Promise.resolve({
        ok,
        json: async () => payload,
        text: async () => JSON.stringify(payload),
    })
}

async function bootSidebar() {
    await import('../components/sidebar.js')
    document.dispatchEvent(new Event('DOMContentLoaded'))
    return window.sidebar
}

describe('Sidebar intrinsic experiment builder', () => {
    beforeEach(() => {
        jest.resetModules()
        jest.useRealTimers()
        injectHtml()
        localStorage.clear()
        window.fetch = jest.fn()
    })

    test('serializes intrinsic manifest from sidebar controls', async () => {
        const sidebar = await bootSidebar()

        document.getElementById('env-width').value = '120'
        document.getElementById('env-height').value = '80'
        document.getElementById('init-resources').value = '900'
        document.getElementById('exp-name').value = 'intrinsic-test-run'
        document.getElementById('exp-steps').value = '600'
        document.getElementById('exp-snapshot-interval').value = '30'
        document.getElementById('exp-selection-pressure').value = 'high'
        document.getElementById('exp-enable-speciation').checked = false

        const manifest = sidebar.buildIntrinsicManifest()
        expect(manifest.schema_version).toBe(1)
        expect(manifest.experiment_type).toBe('intrinsic_evolution')
        expect(manifest.experiment_name).toBe('intrinsic-test-run')
        expect(manifest.base_simulation_config).toEqual({
            'environment.width': 120,
            'environment.height': 80,
            'resources.initial_resources': 900,
        })
        expect(manifest.experiment_config).toMatchObject({
            num_steps: 600,
            snapshot_interval: 30,
            policy: { selection_pressure: 'high' },
            speciation: { enabled: false },
        })
    })

    test('deserializes saved preset back into controls', async () => {
        const sidebar = await bootSidebar()
        localStorage.setItem(
            'intrinsic-dashboard-preset',
            JSON.stringify({
                experiment_name: 'loaded-preset',
                experiment_config: {
                    num_steps: 777,
                    snapshot_interval: 11,
                    policy: { selection_pressure: 'low' },
                    speciation: { enabled: true },
                },
            })
        )

        sidebar.loadExperimentPreset()

        expect(document.getElementById('exp-name').value).toBe('loaded-preset')
        expect(document.getElementById('exp-steps').value).toBe('777')
        expect(document.getElementById('exp-snapshot-interval').value).toBe('11')
        expect(document.getElementById('exp-selection-pressure').value).toBe('low')
        expect(document.getElementById('exp-enable-speciation').checked).toBe(true)
        expect(document.getElementById('exp-status').textContent).toContain('Loaded saved experiment preset.')
    })

    test('validate action posts manifest and shows success status', async () => {
        const sidebar = await bootSidebar()
        window.fetch.mockImplementation((url) => {
            if (url.endsWith('/api/experiments/manifests/validate')) {
                return mockJsonResponse({ status: 'success', data: { is_valid: true } })
            }
            throw new Error(`Unexpected fetch url: ${url}`)
        })

        await sidebar.validateExperimentManifest()

        expect(window.fetch).toHaveBeenCalledTimes(1)
        const [url, options] = window.fetch.mock.calls[0]
        expect(url).toContain('/api/experiments/manifests/validate')
        const body = JSON.parse(options.body)
        expect(body.manifest.experiment_type).toBe('intrinsic_evolution')
        expect(document.getElementById('exp-status').textContent).toBe('Manifest is valid.')
    })

    test('run action updates lifecycle status pending running completed', async () => {
        jest.useFakeTimers()
        const sidebar = await bootSidebar()

        let statusPollCount = 0
        window.fetch.mockImplementation((url) => {
            if (url.endsWith('/api/experiments/run')) {
                return mockJsonResponse({ status: 'accepted', run_id: 'run123' })
            }
            if (url.endsWith('/api/experiments/run123/status')) {
                statusPollCount += 1
                if (statusPollCount === 1) {
                    return mockJsonResponse({ status: 'success', data: { status: 'pending' } })
                }
                if (statusPollCount === 2) {
                    return mockJsonResponse({ status: 'success', data: { status: 'running' } })
                }
                return mockJsonResponse({ status: 'success', data: { status: 'completed' } })
            }
            throw new Error(`Unexpected fetch url: ${url}`)
        })

        await sidebar.runExperimentFromBuilder()
        expect(document.getElementById('exp-status').textContent).toBe('Run run123 started.')

        await jest.advanceTimersByTimeAsync(2000)
        expect(document.getElementById('exp-status').textContent).toBe('Run run123: pending')

        await jest.advanceTimersByTimeAsync(2000)
        expect(document.getElementById('exp-status').textContent).toBe('Run run123: running')

        await jest.advanceTimersByTimeAsync(2000)
        expect(document.getElementById('exp-status').textContent).toBe('Run run123: completed')
    })

    test('new simulation control remains wired (regression)', async () => {
        const sidebar = await bootSidebar()
        window.fetch.mockImplementation((url) => {
            if (url.endsWith('/api/simulation/new')) {
                return mockJsonResponse({ status: 'accepted', sim_id: 'sim123', message: 'Simulation started' })
            }
            throw new Error(`Unexpected fetch url: ${url}`)
        })

        document.getElementById('env-width').value = '140'
        document.getElementById('env-height').value = '90'
        document.getElementById('init-resources').value = '700'
        await sidebar.createNewSimulation()

        expect(window.fetch).toHaveBeenCalledTimes(1)
        const [url, options] = window.fetch.mock.calls[0]
        expect(url).toContain('/api/simulation/new')
        expect(JSON.parse(options.body)).toEqual({
            width: 140,
            height: 90,
            initialResources: 700,
        })
    })
})

