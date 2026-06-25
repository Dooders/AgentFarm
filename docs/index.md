---
layout: home
title: AgentFarm
description: A Python-first platform for agent-based simulation, reinforcement learning, and emergent-behavior research.
---

<section class="hero">
  <div class="container hero__inner">
    <div class="hero__copy">
      <h1 class="hero__title">
        AgentFarm — a research workbench for agent-based simulation.
      </h1>
      <p class="hero__lede">
        Python-first tools for agent-based modeling, reinforcement learning,
        and emergent-behavior research. Build environments, run experiments,
        and persist structured data for reproducible analysis.
      </p>
      <div class="hero__cta">
        <a class="btn btn--primary" href="{{ '/README/' | relative_url }}">Documentation</a>
        <a class="btn" href="https://github.com/Dooders/AgentFarm" target="_blank" rel="noopener">GitHub</a>
        <a class="btn btn--ghost" href="#quickstart">Quick start</a>
      </div>
      <div class="hero__meta">
        <span><strong>Python 3.10+</strong></span>
        <span><strong>PyTorch + Tianshou</strong></span>
        <span><strong>SQLite-backed runs</strong></span>
        <span><strong>structlog</strong></span>
      </div>
    </div>

    <aside class="codeblock" aria-hidden="true">
      <div class="codeblock__head">
        <span>run_simulation.py</span>
        <span>python</span>
      </div>
<pre class="codeblock__body"><span class="tk-kw">from</span> farm.config <span class="tk-kw">import</span> SimulationConfig
<span class="tk-kw">from</span> farm.core.simulation <span class="tk-kw">import</span> run_simulation

config = SimulationConfig.from_centralized_config(
    environment=<span class="tk-str">"development"</span>
)

<span class="tk-cm"># Run, log, and persist a .db for later analysis</span>
env = run_simulation(
    num_steps=config.max_steps,
    config=config,
    path=<span class="tk-str">"simulations"</span>,
)
</pre>
    </aside>
  </div>
</section>

<section class="section" id="features">
  <div class="container">
    <header class="section__head">
      <h2 class="section__title">What's in the box</h2>
      <p class="section__lede">
        Composable primitives for simulation, learning, and analysis — wired together
        with structured logging and reproducible run artifacts.
      </p>
    </header>

    <div class="feature-grid">
      <div class="feature">
        <h3 class="feature__title">Agent-based modeling</h3>
        <p class="feature__desc">
          Compose adaptive agents, environments, and rules. Track interactions,
          resources, and emergent dynamics over time.
        </p>
        <a class="feature__link" href="{{ '/concepts/module-overview/' | relative_url }}">Architecture</a>
      </div>

      <div class="feature">
        <h3 class="feature__title">Reinforcement learning</h3>
        <p class="feature__desc">
          PyTorch + Tianshou-backed decision modules with prioritized experience
          replay and evolvable hyperparameters.
        </p>
        <a class="feature__link" href="{{ '/concepts/deep-q-learning/' | relative_url }}">Deep Q-learning</a>
      </div>

      <div class="feature">
        <h3 class="feature__title">Hyperparameter evolution</h3>
        <p class="feature__desc">
          Genetics-inspired chromosomes for learning hyperparameters with mutation,
          crossover, and adaptive rates.
        </p>
        <a class="feature__link" href="{{ '/design/hyperparameter_chromosome/' | relative_url }}">Chromosomes</a>
      </div>

      <div class="feature">
        <h3 class="feature__title">Spatial indexing</h3>
        <p class="feature__desc">
          KD-tree, Quadtree, and Spatial Hash Grid backends with dirty-region
          tracking — thousands of agents, fast.
        </p>
        <a class="feature__link" href="{{ '/concepts/spatial/spatial_indexing/' | relative_url }}">Spatial indexing</a>
      </div>

      <div class="feature">
        <h3 class="feature__title">Data &amp; analysis</h3>
        <p class="feature__desc">
          Repository-backed databases, behavioral clustering, causal analysis,
          and experiment-level comparisons.
        </p>
        <a class="feature__link" href="{{ '/reference/data/data_api/' | relative_url }}">Data API</a>
        <a class="feature__link" href="{{ '/guides/genetics-analysis/' | relative_url }}">Genetics</a>
      </div>

      <div class="feature">
        <h3 class="feature__title">Structured logging</h3>
        <p class="feature__desc">
          <code>structlog</code>-powered, context-rich, machine-readable logs
          with sampling and sensitive-data censoring.
        </p>
        <a class="feature__link" href="{{ '/reference/logging-quick-reference/' | relative_url }}">Logging</a>
      </div>
    </div>
  </div>
</section>

<section class="section" id="quickstart">
  <div class="container">
    <header class="section__head">
      <h2 class="section__title">Quick start</h2>
      <p class="section__lede">
        Clone the repository, create a virtualenv, install AgentFarm, and run
        your first simulation.
      </p>
    </header>

    <div class="quickstart">
      <ol class="quickstart__steps">
        <li class="quickstart__step">
          <div>
            <h4>Clone &amp; create a virtualenv</h4>
            <p>AgentFarm targets Python 3.10+. A virtualenv keeps experiment dependencies isolated.</p>
          </div>
        </li>
        <li class="quickstart__step">
          <div>
            <h4>Install in editable mode</h4>
            <p>Editable installs make it easy to extend agents, environments, or analysis pipelines.</p>
          </div>
        </li>
        <li class="quickstart__step">
          <div>
            <h4>Run a simulation</h4>
            <p>The CLI emits a SQLite <code>.db</code> under <code>simulations/</code> for downstream analysis.</p>
          </div>
        </li>
        <li class="quickstart__step">
          <div>
            <h4>Open the docs</h4>
            <p>Browse <a href="{{ '/getting-started/installation/' | relative_url }}">getting started</a> and the <a href="{{ '/README/' | relative_url }}">documentation hub</a>.</p>
          </div>
        </li>
      </ol>

      <div class="codeblock">
        <div class="codeblock__head">
          <span>shell</span>
          <span>bash</span>
        </div>
<pre class="codeblock__body"><span class="tk-cm"># 1. Clone</span>
<span class="tk-fn">git</span> <span class="tk-kw">clone</span> <span class="tk-str">https://github.com/Dooders/AgentFarm.git</span>
<span class="tk-fn">cd</span> AgentFarm

<span class="tk-cm"># 2. Virtualenv + install</span>
<span class="tk-fn">python</span> <span class="tk-kw">-m</span> <span class="tk-fn">venv</span> venv
<span class="tk-fn">source</span> venv/bin/activate
<span class="tk-fn">pip</span> <span class="tk-kw">install</span> <span class="tk-kw">-r</span> requirements.txt
<span class="tk-fn">pip</span> <span class="tk-kw">install</span> <span class="tk-kw">-e</span> .

<span class="tk-cm"># 3. Run a small simulation</span>
<span class="tk-fn">python</span> run_simulation.py \
  <span class="tk-kw">--environment</span> <span class="tk-str">development</span> \
  <span class="tk-kw">--steps</span> <span class="tk-num">500</span>

<span class="tk-cm"># 4. Run the test suite</span>
<span class="tk-fn">pytest</span> <span class="tk-kw">-q</span>
</pre>
      </div>
    </div>
  </div>
</section>

<section class="section section--last">
  <div class="container">
    <header class="section__head">
      <h2 class="section__title">From the devlog</h2>
      <p class="section__lede">
        Build notes, design decisions, and experiment outcomes from the AgentFarm team.
      </p>
    </header>

    <ul class="posts">
      <li>
        <a class="post-card" href="{{ '/research/devlog/2026-04-23-evolving-hyperparameter-genomes-foraging-learning-agents/' | relative_url }}">
          <span class="post-card__date">2026-04-23</span>
          <h3 class="post-card__title">Evolving hyperparameter genomes in foraging and learning agents</h3>
          <p class="post-card__excerpt">
            Each agent carries its own hyperparameter chromosome, offspring inherit it (with mutation and
            crossover), and selection is whatever the resource environment happens to apply.
          </p>
        </a>
      </li>
      <li>
        <a class="post-card" href="{{ '/research/devlog/2026-04-17-dna-hyperparameter-evolution/' | relative_url }}">
          <span class="post-card__date">2026-04-17</span>
          <h3 class="post-card__title">DNA-style hyperparameter evolution results</h3>
          <p class="post-card__excerpt">
            Design and initial outcomes of the genetics-inspired hyperparameter evolution work in AgentFarm.
          </p>
        </a>
      </li>
    </ul>

    <p class="section__more">
      <a class="btn" href="{{ '/research/devlog/' | relative_url }}">All devlog posts</a>
    </p>
  </div>
</section>
