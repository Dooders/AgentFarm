;(function() {
	const BASE_URL = 'http://localhost:5000'

	async function fetchSchema() {
		const url = `${BASE_URL}/config/schema`
		const res = await fetch(url, { method: 'GET' })
		if (!res.ok) throw new Error(`Schema request failed: ${res.status}`)
		return await res.json()
	}

	window.configSchemaService = {
		fetchSchema,
	}
})()

