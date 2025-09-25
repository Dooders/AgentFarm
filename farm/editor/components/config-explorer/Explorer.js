/* Config Explorer - Dynamic Forms, Validation, YAML Preview */

;(function() {
	class ConfigExplorer {
		constructor(rootElement) {
			this.root = rootElement
			this.state = {
				sections: [],
				selectedSectionKey: null,
				schema: null,
				config: null,
				lastSavedConfig: null,
				unsaved: false,
				fieldErrors: {}, // { sectionKey: { fieldName: [messages] } }
			}
			this._validateTimeout = null
			this.init()
		}

		async init() {
			this.root.className = 'explorer-root'
			this.root.innerHTML = ''
			this.buildLayout()
			await this.loadSchema()
			await this.loadInitialConfig()
			this.renderSections()
			this.selectFirstSection()
			this.updateYamlPreview()
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
					<span id="unsaved-indicator" class="unsaved-indicator" style="display:none;"><span class="dot"></span> Unsaved</span>
				</div>
				<div class="right">
					<input id="save-path" class="search" placeholder="Save path (optional)" />
					<button id="save-config" class="btn">Save</button>
				</div>
			`
			bar.querySelector('#back-to-legacy').onclick = () => window.hideConfigExplorer()
			bar.querySelector('#save-config').onclick = () => this.onSave()
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
				const sections = Object.entries((res && res.sections) || {}).map(([key, meta]) => ({ key, title: (meta && meta.title) || key }))
				this.state.sections = sections
			} catch (err) {
				console.error('Failed to load schema', err)
				this.sectionListEl.innerHTML = '<div class="error">Failed to load schema.</div>'
			}
		}

		async loadInitialConfig() {
			try {
				const res = await window.configSchemaService.loadConfig(undefined)
				if (res && res.success && res.config) {
					this.state.config = res.config
					this.state.lastSavedConfig = JSON.parse(JSON.stringify(res.config))
					this.state.unsaved = false
					this.updateUnsavedIndicator()
					return
				}
			} catch (e) {
				console.warn('Failed to load existing config, falling back to defaults')
			}
			// Seed defaults from schema if load failed
			this.state.config = this.buildDefaultConfigFromSchema()
			this.state.lastSavedConfig = JSON.parse(JSON.stringify(this.state.config))
			this.state.unsaved = false
			this.updateUnsavedIndicator()
		}

		buildDefaultConfigFromSchema() {
			const combined = {}
			const sections = this.state.schema?.sections || {}
			// Top-level (simulation) fields live at root
			const simProps = sections.simulation?.properties || {}
			Object.entries(simProps).forEach(([name, meta]) => {
				combined[name] = this.cloneDefault(meta.default)
			})
			// Nested sections
			;['visualization','redis','observation'].forEach((sec) => {
				const props = sections[sec]?.properties || {}
				combined[sec] = {}
				Object.entries(props).forEach(([name, meta]) => {
					combined[sec][name] = this.cloneDefault(meta.default)
				})
			})
			return combined
		}

		cloneDefault(val) {
			if (Array.isArray(val)) return val.map((v) => this.cloneDefault(v))
			if (val && typeof val === 'object') return JSON.parse(JSON.stringify(val))
			return val
		}

		renderSections() {
			const list = document.createElement('div')
			list.className = 'sections'
			this.state.sections.forEach((s) => {
				const item = document.createElement('button')
				item.className = 'section-item'
				item.textContent = s.title
				item.setAttribute('data-key', s.key)
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
				el.classList.toggle('active', el.getAttribute('data-key') === key)
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
				row.setAttribute('data-field', name)
				const label = document.createElement('label')
				label.textContent = name
				const control = this.buildInputControl(key, name, p)
				const msg = document.createElement('div')
				msg.className = 'validation-msg'
				row.appendChild(label)
				row.appendChild(control)
				row.appendChild(msg)
				form.appendChild(row)
			})
			container.innerHTML = ''
			container.appendChild(form)
		}

		buildInputControl(sectionKey, fieldName, meta) {
			const type = meta.type || 'string'
			const hasEnum = Array.isArray(meta.enum)
			let el
			const currentValue = this.getFieldValue(sectionKey, fieldName)
			if (hasEnum) {
				el = document.createElement('select')
				meta.enum.forEach((opt) => {
					const o = document.createElement('option')
					o.value = String(opt)
					o.textContent = String(opt)
					if (String(currentValue) === String(opt)) o.selected = true
					el.appendChild(o)
				})
				el.onchange = (e) => this.onFieldChange(sectionKey, fieldName, e.target.value, meta)
				return el
			}
			if (type === 'boolean') {
				el = document.createElement('input')
				el.type = 'checkbox'
				el.checked = Boolean(currentValue)
				el.onchange = (e) => this.onFieldChange(sectionKey, fieldName, Boolean(e.target.checked), meta)
				return el
			}
			if (type === 'integer' || type === 'number') {
				el = document.createElement('input')
				el.type = 'number'
				if (meta.minimum !== undefined) el.min = String(meta.minimum)
				if (meta.maximum !== undefined) el.max = String(meta.maximum)
				el.step = type === 'integer' ? '1' : 'any'
				if (currentValue !== undefined && currentValue !== null) el.value = String(currentValue)
				el.oninput = (e) => this.onFieldChange(sectionKey, fieldName, e.target.value, meta)
				return el
			}
			if (type === 'object' || type === 'array') {
				el = document.createElement('textarea')
				el.placeholder = type === 'array' ? '[...] JSON' : '{...} JSON'
				el.rows = 3
				try { el.value = currentValue != null ? JSON.stringify(currentValue, null, 2) : '' } catch (_) { el.value = '' }
				el.oninput = (e) => this.onFieldChange(sectionKey, fieldName, e.target.value, meta)
				return el
			}
			// default string
			el = document.createElement('input')
			el.type = 'text'
			el.value = currentValue == null ? '' : String(currentValue)
			el.placeholder = meta.type || 'string'
			el.oninput = (e) => this.onFieldChange(sectionKey, fieldName, e.target.value, meta)
			return el
		}

		getFieldValue(sectionKey, fieldName) {
			if (sectionKey === 'simulation') return this.state.config?.[fieldName]
			return this.state.config?.[sectionKey]?.[fieldName]
		}

		setFieldValue(sectionKey, fieldName, value) {
			if (sectionKey === 'simulation') {
				this.state.config[fieldName] = value
				return
			}
			if (!this.state.config[sectionKey]) this.state.config[sectionKey] = {}
			this.state.config[sectionKey][fieldName] = value
		}

		onFieldChange(sectionKey, fieldName, rawValue, meta) {
			// Coerce value
			let value = rawValue
			if (meta.type === 'integer') {
				value = rawValue === '' ? null : parseInt(rawValue, 10)
				if (Number.isNaN(value)) value = null
			}
			if (meta.type === 'number') {
				value = rawValue === '' ? null : parseFloat(rawValue)
				if (Number.isNaN(value)) value = null
			}
			if (meta.type === 'object' || meta.type === 'array') {
				try {
					value = rawValue === '' ? (meta.type === 'array' ? [] : {}) : JSON.parse(rawValue)
					this.setFieldError(sectionKey, fieldName, null)
				} catch (e) {
					this.setFieldError(sectionKey, fieldName, ['Invalid JSON'])
					return
				}
			}

			// Client-side validation for bounds
			const clientErrors = []
			if ((meta.type === 'integer' || meta.type === 'number') && value != null) {
				if (meta.minimum !== undefined && value < meta.minimum) clientErrors.push(`Must be ≥ ${meta.minimum}`)
				if (meta.maximum !== undefined && value > meta.maximum) clientErrors.push(`Must be ≤ ${meta.maximum}`)
			}
			this.setFieldError(sectionKey, fieldName, clientErrors.length ? clientErrors : null)

			// Update state
			this.setFieldValue(sectionKey, fieldName, value)
			this.markUnsaved(true)
			this.updateYamlPreview()

			// Debounced server validation
			clearTimeout(this._validateTimeout)
			this._validateTimeout = setTimeout(() => this.runServerValidation(), 350)
		}

		setFieldError(sectionKey, fieldName, messages) {
			if (!this.state.fieldErrors[sectionKey]) this.state.fieldErrors[sectionKey] = {}
			if (!messages || messages.length === 0) {
				delete this.state.fieldErrors[sectionKey][fieldName]
			} else {
				this.state.fieldErrors[sectionKey][fieldName] = messages
			}
			// Update UI on the active section if applicable
			if (this.state.selectedSectionKey === sectionKey) {
				const row = this.detailsEl.querySelector(`.form-row[data-field="${fieldName}"]`)
				if (row) {
					row.classList.toggle('row-error', !!messages)
					const msg = row.querySelector('.validation-msg')
					if (msg) msg.textContent = messages ? messages.join(' ') : ''
				}
			}
		}

		markUnsaved(flag) {
			this.state.unsaved = !!flag
			this.updateUnsavedIndicator()
		}

		updateUnsavedIndicator() {
			const el = this.root.querySelector('#unsaved-indicator')
			if (el) el.style.display = this.state.unsaved ? '' : 'none'
			const saveBtn = this.root.querySelector('#save-config')
			if (saveBtn) saveBtn.disabled = !this.state.unsaved
		}

		async runServerValidation() {
			try {
				const res = await window.configSchemaService.validateConfig(this.state.config)
				this.showGlobalValidation(res && res.success ? 'valid' : 'error', res && res.message ? res.message : '')
			} catch (e) {
				this.showGlobalValidation('error', 'Validation failed')
			}
		}

		showGlobalValidation(status, message) {
			// Use YAML header area to reflect status succinctly
			const header = this.yamlEl.querySelector('.yaml-header')
			if (!header) return
			header.innerHTML = `YAML Preview <span class="badge ${status}">${status === 'valid' ? 'Valid' : 'Invalid'}</span>${message ? ` <span class="note">${message}</span>` : ''}`
		}

		updateYamlPreview() {
			const pre = this.yamlEl.querySelector('.yaml-code')
			if (!pre) return
			pre.textContent = this.toYaml(this.state.config)
		}

		toYaml(obj, indent = 0) {
			const pad = (n) => '  '.repeat(n)
			if (obj === null || obj === undefined) return 'null'
			if (typeof obj !== 'object') {
				if (typeof obj === 'string') {
					if (obj.includes('\n') || obj.includes(':') || obj.includes('#') || obj.trim() !== obj) return JSON.stringify(obj)
					return obj
				}
				return String(obj)
			}
			if (Array.isArray(obj)) {
				if (obj.length === 0) return '[]'
				return obj.map((v) => `${pad(indent)}- ${this.toYaml(v, indent + 1)}`).join('\n')
			}
			const keys = Object.keys(obj)
			if (keys.length === 0) return '{}'
			return keys
				.map((k) => {
					const val = obj[k]
					const isObj = val && typeof val === 'object'
					const rendered = this.toYaml(val, indent + 1)
					if (isObj && !Array.isArray(val)) {
						return `${pad(indent)}${k}:\n${rendered}`
					}
					if (Array.isArray(val)) {
						if (val.length === 0) return `${pad(indent)}${k}: []`
						const arr = val.map((v) => `${pad(indent + 1)}- ${this.toYaml(v, indent + 2)}`).join('\n')
						return `${pad(indent)}${k}:\n${arr}`
					}
					return `${pad(indent)}${k}: ${rendered}`
				})
				.join('\n')
		}

		async onSave() {
			const btn = this.root.querySelector('#save-config')
			if (btn) { btn.disabled = true; btn.textContent = 'Saving…' }
			try {
				const path = (this.root.querySelector('#save-path')?.value || undefined)
				const res = await window.configSchemaService.saveConfig(this.state.config, path)
				if (res && res.success) {
					this.state.lastSavedConfig = JSON.parse(JSON.stringify(res.config || this.state.config))
					this.markUnsaved(false)
					this.showGlobalValidation('valid', 'Configuration saved')
				} else {
					this.showGlobalValidation('error', (res && res.message) || 'Save failed')
				}
			} catch (e) {
				this.showGlobalValidation('error', 'Save failed')
			} finally {
				if (btn) { btn.disabled = !this.state.unsaved; btn.textContent = 'Save' }
			}
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

