;(function() {
	const STORAGE_KEY = 'agentFarm:onboarding:v1'

	class OnboardingOverlay {
		constructor() {
			this.root = null
			this.primaryBtn = null
			this.remindBtn = null
			this.closeBtn = null
			this.initialized = false
			this.isOpen = false
			this.focusableItems = []
		}

		init() {
			if (this.initialized) return
			this.initialized = true

			this.render()
			this.attachEvents()

			if (this.shouldShow()) {
				this.open()
			}
		}

		shouldShow() {
			try {
				return localStorage.getItem(STORAGE_KEY) !== '1'
			} catch (err) {
				return true
			}
		}

		markSeen() {
			try {
				localStorage.setItem(STORAGE_KEY, '1')
			} catch (err) {
				// ignore
			}
		}

		reset() {
			try {
				localStorage.removeItem(STORAGE_KEY)
			} catch (err) {
				// ignore
			}
		}

		render() {
			if (this.root) return

			const overlay = document.createElement('div')
			overlay.className = 'onboarding-overlay'
			overlay.setAttribute('role', 'dialog')
			overlay.setAttribute('aria-modal', 'true')
			overlay.setAttribute('aria-labelledby', 'onboarding-title')
			overlay.innerHTML = `
				<div class="onboarding-card">
					<button type="button" class="onboarding-close" aria-label="Dismiss welcome overlay">
						<span aria-hidden="true">&times;</span>
					</button>
					<div class="onboarding-header">
						<div class="onboarding-kicker">Agent Farm Studio · Builder Fellow Preview</div>
						<h1 id="onboarding-title">Welcome to the Agent Farm Studio</h1>
						<p class="onboarding-lead">
							This workspace is where we explore autonomous ecosystems: launch simulations, tune scenarios, and study how agents learn to cooperate or compete.
						</p>
					</div>
					<div class="onboarding-grid">
						<section class="onboarding-section">
							<h2>What ships today</h2>
							<ul>
								<li>
									<strong>Interactive Simulation Loop</strong>
									<span>Kick off runs via the REST API, stream every step over WebSockets, and inspect agents with canvas overlays plus live population and resource charts.</span>
								</li>
								<li>
									<strong>Config Explorer (Preview)</strong>
									<span>Schema-driven forms with diffing, presets, validation, and YAML export keep experiments reproducible while we iterate quickly.</span>
								</li>
								<li>
									<strong>Analysis Toolkit</strong>
									<span>Structured logging, benchmark harnesses, and analyzer modules provide the evidence we need to explain emergent behavior.</span>
								</li>
							</ul>
						</section>
						<section class="onboarding-section">
							<h2>Where we're headed next</h2>
							<ul>
								<li>
									<strong>Phase 1 · Modular Core</strong>
									<span>Refactoring the environment into focused managers with dependency injection and a plugin-ready action registry (<a href="https://github.com/Dooders/AgentFarm/blob/main/docs/architectural_recommendations.md#1-modularity-recommendations" target="_blank" rel="noreferrer">roadmap</a>).</span>
								</li>
								<li>
									<strong>Phase 2 · Extensibility</strong>
									<span>Centralized configuration management, extension registries, and a richer event bus so teams can drop in custom agents, analyzers, and tooling.</span>
								</li>
								<li>
									<strong>Phase 3 · Builder Experience</strong>
									<span>Multi-run comparisons, ML-assisted agent coaching, and packaged desktop builds to make the studio a turnkey lab for the next cohort.</span>
								</li>
							</ul>
						</section>
					</div>
					<div class="onboarding-footer">
						<div class="onboarding-actions">
							<button type="button" class="onboarding-primary" data-onboarding-focus>Start exploring</button>
							<button type="button" class="onboarding-ghost">Remind me later</button>
						</div>
						<div class="onboarding-links">
							<a href="https://github.com/Dooders/AgentFarm#readme" target="_blank" rel="noreferrer">Project overview</a>
							<a href="https://github.com/Dooders/AgentFarm/tree/main/docs" target="_blank" rel="noreferrer">Documentation index</a>
						</div>
					</div>
				</div>
			`

			document.body.appendChild(overlay)

			this.root = overlay
			this.primaryBtn = overlay.querySelector('.onboarding-primary')
			this.remindBtn = overlay.querySelector('.onboarding-ghost')
			this.closeBtn = overlay.querySelector('.onboarding-close')
		}

		open(force) {
			if (!force && !this.shouldShow()) return
			if (!this.root) this.render()

			this.root.classList.add('active')
			document.body.classList.add('onboarding-open')
			this.isOpen = true
			this.updateFocusableItems()

			const focusTarget = this.root.querySelector('[data-onboarding-focus]') || this.focusableItems[0]
			if (focusTarget && typeof focusTarget.focus === 'function') {
				focusTarget.focus()
			}
		}

		close({ markSeen = true } = {}) {
			if (!this.root) return

			if (markSeen) this.markSeen()

			this.root.classList.remove('active')
			document.body.classList.remove('onboarding-open')
			this.isOpen = false
		}

		updateFocusableItems() {
			if (!this.root) {
				this.focusableItems = []
				return
			}

			const selector = [
				'button:not([disabled])',
				'a[href]',
				'input:not([disabled])',
				'select:not([disabled])',
				'textarea:not([disabled])',
				'[tabindex]:not([tabindex="-1"])'
			].join(',')

			this.focusableItems = Array.from(this.root.querySelectorAll(selector))
		}

		handleKeyDown(event) {
			if (!this.isOpen) return

			if (event.key === 'Escape') {
				event.preventDefault()
				this.close({ markSeen: true })
				return
			}

			if (event.key === 'Tab') {
				if (!this.focusableItems.length) return
				const first = this.focusableItems[0]
				const last = this.focusableItems[this.focusableItems.length - 1]
				const active = document.activeElement

				if (event.shiftKey) {
					if (active === first || !this.root.contains(active)) {
						event.preventDefault()
						last.focus()
					}
				} else {
					if (active === last) {
						event.preventDefault()
						first.focus()
					}
				}
			}
		}

		attachEvents() {
			if (!this.root || this._eventsAttached) return
			this._eventsAttached = true

			this.primaryBtn?.addEventListener('click', () => {
				this.close({ markSeen: true })
			})

			this.remindBtn?.addEventListener('click', () => {
				this.close({ markSeen: false })
			})

			this.closeBtn?.addEventListener('click', () => {
				this.close({ markSeen: true })
			})

			this.root.addEventListener('keydown', (event) => this.handleKeyDown(event))
		}
	}

	const overlay = new OnboardingOverlay()

	function boot() {
		overlay.init()
	}

	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', boot)
	} else {
		boot()
	}

	if (typeof window !== 'undefined') {
		window.AgentFarmOnboarding = {
			show: (force) => overlay.open(force !== undefined ? force : true),
			reset: () => overlay.reset()
		}
	}
})()

