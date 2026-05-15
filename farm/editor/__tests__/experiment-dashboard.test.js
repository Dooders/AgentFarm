/** @jest-environment jsdom */

function injectHtml() {
    document.body.innerHTML = '<div id="stats">Simulation stats root</div><div id="experiment-dashboard"></div>'
}

function mockJsonResponse(payload, ok = true, status = 200) {
    return Promise.resolve({
        ok,
        status,
        json: async () => payload,
    })
}

async function bootDashboard() {
    await import('../components/experiment-dashboard.js')
    document.dispatchEvent(new Event('DOMContentLoaded'))
    return window.experimentDashboard
}

describe('ExperimentDashboard', () => {
    beforeEach(() => {
        jest.resetModules()
        injectHtml()
        window.fetch = jest.fn()
    })

    test('loads run views and renders summary cards payload', async () => {
        const dashboard = await bootDashboard()

        window.fetch
            .mockImplementationOnce(() => mockJsonResponse({
                status: 'success',
                data: [{ view_id: 'summary_cards', title: 'Run Summary' }],
            }))
            .mockImplementationOnce(() => mockJsonResponse({
                status: 'success',
                data: {
                    view_type: 'summary_cards',
                    cards: [{ label: 'Final Population', value: 42 }],
                },
            }))

        await dashboard.loadRun('run123')

        expect(document.getElementById('dashboard-run-id').textContent).toContain('run123')
        expect(document.getElementById('dashboard-body').innerHTML).toContain('Final Population')
        expect(document.getElementById('dashboard-body').innerHTML).toContain('42')
    })

    test('shows empty view list state', async () => {
        await bootDashboard()

        window.fetch.mockImplementationOnce(() => mockJsonResponse({ status: 'success', data: [] }))
        await window.experimentDashboard.loadRun('run-empty')

        expect(document.getElementById('dashboard-body').textContent).toContain('No dashboard views are available')
        expect(document.getElementById('dashboard-view-select').disabled).toBe(true)
        expect(document.getElementById('dashboard-load-view').disabled).toBe(true)
    })

    test('shows error when listing views fails', async () => {
        await bootDashboard()

        window.fetch.mockImplementationOnce(() => mockJsonResponse({ status: 'error' }, false, 500))
        await window.experimentDashboard.loadRun('run-fail-views')

        expect(document.getElementById('dashboard-body').textContent).toContain('Failed to load views: HTTP 500')
    })

    test('shows error when loading selected view fails', async () => {
        await bootDashboard()

        window.fetch
            .mockImplementationOnce(() => mockJsonResponse({
                status: 'success',
                data: [{ view_id: 'summary_cards', title: 'Run Summary' }],
            }))
            .mockImplementationOnce(() => mockJsonResponse({ status: 'error' }, false, 404))

        await window.experimentDashboard.loadRun('run-fail-view-data')
        expect(document.getElementById('dashboard-body').textContent).toContain('Failed to load view data: HTTP 404')
    })

    test('does not overwrite simulation stats container when rendering dashboard', async () => {
        await bootDashboard()
        expect(document.getElementById('stats').textContent).toContain('Simulation stats root')
    })

    test('renders timeseries branch', async () => {
        const dashboard = await bootDashboard()
        dashboard.renderViewData({
            view_type: 'timeseries',
            title: 'Gene trajectories',
            series: [{ label: 'learning_rate mean', values: [0.1, 0.2] }],
        })

        expect(document.getElementById('dashboard-body').innerHTML).toContain('Gene trajectories')
        expect(document.getElementById('dashboard-body').innerHTML).toContain('learning_rate mean')
    })

    test('renders distribution_over_time branch', async () => {
        const dashboard = await bootDashboard()
        dashboard.renderViewData({
            view_type: 'distribution_over_time',
            title: 'Gene distribution history',
            snapshots: [{ step: 100, by_gene: { learning_rate: [0.1, 0.2] } }],
        })

        expect(document.getElementById('dashboard-body').innerHTML).toContain('Gene distribution history')
        expect(document.getElementById('dashboard-body').innerHTML).toContain('100')
    })

    test('renders lineage_or_clusters branch and tolerates optional data', async () => {
        const dashboard = await bootDashboard()
        dashboard.renderViewData({
            view_type: 'lineage_or_clusters',
            title: 'Speciation index',
            timeseries: { series: [{ values: [0.13] }] },
            clusters: [],
        })

        const html = document.getElementById('dashboard-body').innerHTML
        expect(html).toContain('Speciation index')
        expect(html).toContain('0.13')
        expect(html).toContain('Cluster Records')
        expect(html).toContain('0')
    })

    test('renders no renderer fallback for unknown view types', async () => {
        const dashboard = await bootDashboard()
        dashboard.renderViewData({ view_type: 'unknown_renderer' })
        expect(document.getElementById('dashboard-body').textContent).toContain('No renderer available')
    })

    test('renders empty-state messages for branch payload gaps', async () => {
        const dashboard = await bootDashboard()

        dashboard.renderViewData({ view_type: 'summary_cards', cards: [] })
        expect(document.getElementById('dashboard-body').textContent).toContain('No summary cards')

        dashboard.renderViewData({ view_type: 'timeseries', series: [] })
        expect(document.getElementById('dashboard-body').textContent).toContain('No time series data')

        dashboard.renderViewData({ view_type: 'distribution_over_time', snapshots: [] })
        expect(document.getElementById('dashboard-body').textContent).toContain('No distribution snapshots')
    })

    test('renders adapter values as text content to avoid HTML injection', async () => {
        const dashboard = await bootDashboard()
        dashboard.renderViewData({
            view_type: 'summary_cards',
            cards: [{ label: '<img src=x onerror=alert(1)>', value: '<b>42</b>' }],
        })

        const body = document.getElementById('dashboard-body')
        expect(body.innerHTML).toContain('&lt;img src=x onerror=alert(1)&gt;')
        expect(body.innerHTML).toContain('&lt;b&gt;42&lt;/b&gt;')
        expect(body.querySelector('img')).toBeNull()
        expect(body.querySelector('b')).toBeNull()
    })
})
