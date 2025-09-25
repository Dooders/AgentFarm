/* Config Explorer Scaffold - Phase 1 */

;(function() {
	class ConfigExplorer {
		constructor(rootElement) {
			this.root = rootElement
			this.state = { sections: [], selectedSectionKey: null, schema: null }
			this.init()
		}

		async init() {
			this.root.className = 'explorer-root'
			this.root.innerHTML = ''
			this.buildLayout()
			await this.loadSchema()
			this.renderSections()
			this.selectFirstSection()
		}

		buildLayout() {
			const lrContainer = document.createElement('div')
			lrContainer.className = 'split-h'

			this.sectionListEl = document.createElement('div')
			this.sectionListEl.className = 'section-list'

			const resizerV = document.createElement('div')
			resizerV.className = 'resizer-v'

			const rightPane = document.createElement('div')
			rightPane.className = 'right-pane'

			const tbContainer = document.createElement('div')
			tbContainer.className = 'split-v'

			this.detailsEl = document.createElement('div')
			this.detailsEl.className = 'details-panel'

			const resizerH = document.createElement('div')
			resizerH.className = 'resizer-h'

			this.yamlEl = document.createElement('div')
			this.yamlEl.className = 'yaml-panel'

			this.detailsEl.innerHTML = `
				<div class="details-header">
					<div class="title" id="details-title">Configuration</div>
					<div class="docs-links">
						<a href="https://github.com/Dooders/AgentFarm/blob/main/docs/configuration_guide.md" target="_blank" rel="noreferrer">Guide</a>
						<a href="https://github.com/Dooders/AgentFarm/blob/main/docs/api_reference.md" target="_blank" rel="noreferrer">API</a>
					</div>
				</div>
				<div class="details-content" id="details-content"></div>
			`

			this.yamlEl.innerHTML = `
				<div class="yaml-header">YAML Preview (placeholder)</div>
				<pre class="yaml-code"># YAML preview will appear here as you edit settings.\n\n# Phase 2 will render live YAML.</pre>
			`

			tbContainer.appendChild(this.detailsEl)
			tbContainer.appendChild(resizerH)
			tbContainer.appendChild(this.yamlEl)
			rightPane.appendChild(tbContainer)

			lrContainer.appendChild(this.sectionListEl)
			lrContainer.appendChild(resizerV)
			lrContainer.appendChild(rightPane)

			this.root.appendChild(this.buildToolbar())
			this.root.appendChild(lrContainer)

			this.enableVerticalResizer(resizerV, this.sectionListEl, rightPane)
			this.enableHorizontalResizer(resizerH, this.detailsEl, this.yamlEl)
		}

		buildToolbar() {
			const bar = document.createElement('div')
			bar.className = 'explorer-toolbar'
			bar.innerHTML = `
				<div class="left">
					<button id="back-to-legacy" class="btn">← Back</button>
					<span class="sep"></span>
					<strong>Config Explorer</strong>
				</div>
				<div class="right">
					<input id="explorer-search" class="search" placeholder="Search (placeholder)" />
				</div>
			`
			bar.querySelector('#back-to-legacy').onclick = () => window.hideConfigExplorer()
			return bar
		}

		enableVerticalResizer(resizer, leftEl, rightEl) {
			let isDragging = false
			resizer.addEventListener('mousedown', () => {
				isDragging = true
				document.body.classList.add('resizing')
			})
			document.addEventListener('mousemove', (e) => {
				if (!isDragging) return
				const containerRect = this.root.getBoundingClientRect()
				const minLeft = 200
				const maxLeft = Math.max(300, containerRect.width - 400)
				let newLeft = e.clientX - containerRect.left
				newLeft = Math.max(minLeft, Math.min(maxLeft, newLeft))
				leftEl.style.width = `${newLeft}px`
				rightEl.style.width = `calc(100% - ${newLeft}px - 6px)`
			})
			document.addEventListener('mouseup', () => {
				isDragging = false
				document.body.classList.remove('resizing')
			})
		}

		enableHorizontalResizer(resizer, topEl, bottomEl) {
			let isDragging = false
			resizer.addEventListener('mousedown', () => {
				isDragging = true
				document.body.classList.add('resizing')
			})
			document.addEventListener('mousemove', (e) => {
				if (!isDragging) return
				const rect = topEl.parentElement.getBoundingClientRect()
				const minTop = 200
				const maxTop = Math.max(260, rect.height - 200)
				let newTop = e.clientY - rect.top
				newTop = Math.max(minTop, Math.min(maxTop, newTop))
				topEl.style.height = `${newTop}px`
				bottomEl.style.height = `calc(100% - ${newTop}px - 6px)`
			})
			document.addEventListener('mouseup', () => {
				isDragging = false
				document.body.classList.remove('resizing')
			})
		}

		async loadSchema() {
			try {
				const res = await window.configSchemaService.fetchSchema()
				this.state.schema = res
				const sections = Object.entries(res.sections || {}).map(([key, meta]) => ({ key, title: meta.title || key }))
				this.state.sections = sections
			} catch (err) {
				console.error('Failed to load schema', err)
				this.sectionListEl.innerHTML = '<div class="error">Failed to load schema.</div>'
			}
		}

		renderSections() {
			const list = document.createElement('div')
			list.className = 'sections'
			this.state.sections.forEach((s) => {
				const item = document.createElement('button')
				item.className = 'section-item'
				item.textContent = s.title
				item.onclick = () => this.onSelectSection(s.key)
				if (s.key === this.state.selectedSectionKey) item.classList.add('active')
				list.appendChild(item)
			})

			this.sectionListEl.innerHTML = ''
			const header = document.createElement('div')
			header.className = 'section-header'
			header.textContent = 'Sections'
			this.sectionListEl.appendChild(header)
			this.sectionListEl.appendChild(list)
		}

		selectFirstSection() {
			if (this.state.sections.length > 0) this.onSelectSection(this.state.sections[0].key)
		}

		onSelectSection(key) {
			this.state.selectedSectionKey = key
			Array.from(this.sectionListEl.querySelectorAll('.section-item')).forEach((el) => {
				el.classList.toggle('active', el.textContent === (this.state.schema.sections[key].title || key))
			})
			const titleEl = this.detailsEl.querySelector('#details-title')
			if (titleEl) titleEl.textContent = this.state.schema.sections[key].title || key
			this.renderDetailsContent(key)
		}

		renderDetailsContent(key) {
			const meta = this.state.schema.sections[key]
			const container = this.detailsEl.querySelector('#details-content')
			if (!meta) {
				container.innerHTML = '<div class="empty">No metadata for section.</div>'
				return
			}
			const props = meta.properties || {}
			const form = document.createElement('div')
			form.className = 'form-scaffold'
			Object.entries(props).forEach(([name, p]) => {
				const row = document.createElement('div')
				row.className = 'form-row'
				const label = document.createElement('label')
				label.textContent = name
				const input = document.createElement('input')
				input.disabled = true
				input.placeholder = p.type ? `${p.type}` : 'value'
				row.appendChild(label)
				row.appendChild(input)
				form.appendChild(row)
			})
			container.innerHTML = ''
			container.appendChild(form)
		}
	}

	function ensureGlobalToggles(configExplorer) {
		window.showConfigExplorer = function() {
			const side = document.getElementById('sidebar')
			const main = document.getElementById('main-content')
			if (side) side.style.display = 'none'
			if (main) main.style.display = 'none'
			configExplorer.root.style.display = 'flex'
		}
		window.hideConfigExplorer = function() {
			const side = document.getElementById('sidebar')
			const main = document.getElementById('main-content')
			if (side) side.style.display = ''
			if (main) main.style.display = ''
			const explorer = document.getElementById('config-explorer')
			if (explorer) explorer.style.display = 'none'
		}
	}

	function boot() {
		const root = document.getElementById('config-explorer')
		if (!root) return
		const explorer = new ConfigExplorer(root)
		ensureGlobalToggles(explorer)
	}

	document.addEventListener('DOMContentLoaded', boot)
})()

