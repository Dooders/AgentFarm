;(function() {
	const BASE_URL = 'http://localhost:5000'

	async function fetchSchema() {
		const url = `${BASE_URL}/config/schema`
		const res = await fetch(url, { method: 'GET' })
		if (!res.ok) throw new Error(`Schema request failed: ${res.status}`)
		return await res.json()
	}

	async function validateConfig(config) {
		const url = `${BASE_URL}/config/validate`
		const res = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ config }),
		})
		if (!res.ok) return { success: false, message: `HTTP ${res.status}` }
		return await res.json()
	}

	async function saveConfig(config, path) {
		const url = `${BASE_URL}/config/save`
		const res = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ config, path }),
		})
		if (!res.ok) return { success: false, message: `HTTP ${res.status}` }
		return await res.json()
	}

	async function loadConfig(path) {
		const url = `${BASE_URL}/config/load`
		const res = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ path }),
		})
		if (!res.ok) return { success: false, message: `HTTP ${res.status}` }
		return await res.json()
	}

	window.configSchemaService = {
		fetchSchema,
		validateConfig,
		saveConfig,
		loadConfig,
	}
})()

