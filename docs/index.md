---
layout: home
title: AgentFarm
description: A Python-first platform for agent-based simulation, reinforcement learning, and emergent-behavior research.
---

<section class="hero">
  <div class="container hero__inner">
    <div class="hero__copy">
      <span class="hero__eyebrow"><span class="dot"></span> In active development · MIT-style research code</span>
      <h1 class="hero__title">
        Simulate agents.<br />
        <span class="gradient">Study what emerges.</span>
      </h1>
      <p class="hero__lede">
        AgentFarm is an open-source workbench for agent-based modeling, reinforcement learning experiments,
        and complex adaptive systems research. Build environments, evolve hyperparameters, and analyze
        emergent behavior at scale.
      </p>
      <div class="hero__cta">
        <a class="btn btn--primary" href="{{ '/README/' | relative_url }}">Read the docs</a>
        <a class="btn btn--ghost" href="https://github.com/Dooders/AgentFarm" target="_blank" rel="noopener">
          GitHub
          <span aria-hidden="true">↗</span>
        </a>
      </div>
      <div class="hero__meta">
        <span><strong>Python 3.10+</strong> · simulation core</span>
        <span><strong>structlog</strong> · observability</span>
        <span><strong>Tianshou + PyTorch</strong> · learning</span>
      </div>
    </div>

    <aside class="hero__card" aria-hidden="true">
      <div class="hero__card-bar">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        <span class="label">run_simulation.py</span>
      </div>
<pre class="hero__code"><span class="tk-kw">from</span> farm.config <span class="tk-kw">import</span> SimulationConfig
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
      <span class="section__eyebrow">What it does</span>
      <h2 class="section__title">A platform built for emergent-behavior research</h2>
      <p class="section__lede">
        Compose simulations, run reinforcement-learning experiments, and analyze what comes out — with
        structured data, reproducible runs, and a clean Python API.
      </p>
    </header>

    <div class="feature-grid">
      <div class="feature">
        <div class="feature__icon">AB</div>
        <h3 class="feature__title">Agent-based modeling</h3>
        <p class="feature__desc">
          Compose adaptive agents, environments, and rules. Track interactions, resources, and emergent dynamics over time.
        </p>
        <a class="feature__link" href="{{ '/features/agent_based_modeling_analysis/' | relative_url }}">Explore modeling</a>
      </div>

      <div class="feature">
        <div class="feature__icon">RL</div>
        <h3 class="feature__title">Reinforcement learning</h3>
        <p class="feature__desc">
          PyTorch + Tianshou-backed decision modules with prioritized experience replay and evolvable hyperparameters.
        </p>
        <a class="feature__link" href="{{ '/deep_q_learning/' | relative_url }}">Deep Q-learning guide</a>
      </div>

      <div class="feature">
        <div class="feature__icon">EV</div>
        <h3 class="feature__title">Hyperparameter evolution</h3>
        <p class="feature__desc">
          Genetics-inspired chromosomes for learning hyperparameters with mutation, crossover, and adaptive rates.
        </p>
        <a class="feature__link" href="{{ '/design/hyperparameter_chromosome/' | relative_url }}">Chromosome design</a>
      </div>

      <div class="feature">
        <div class="feature__icon">SP</div>
        <h3 class="feature__title">Spatial indexing at scale</h3>
        <p class="feature__desc">
          KD-tree, Quadtree, and Spatial Hash Grid backends with dirty-region tracking — thousands of agents, fast.
        </p>
        <a class="feature__link" href="{{ '/features/spatial_indexing_performance/' | relative_url }}">Performance details</a>
      </div>

      <div class="feature">
        <div class="feature__icon">DB</div>
        <h3 class="feature__title">Data &amp; analysis</h3>
        <p class="feature__desc">
          Repository-backed databases, behavioral clustering, causal analysis, and experiment-level comparisons.
        </p>
        <a class="feature__link" href="{{ '/features/data_system/' | relative_url }}">Data system</a>
        <a class="feature__link" href="{{ '/genetics_analysis/' | relative_url }}">Genetics analysis module</a>
      </div>

      <div class="feature">
        <div class="feature__icon">LO</div>
        <h3 class="feature__title">Structured logging</h3>
        <p class="feature__desc">
          <code>structlog</code>-powered context-rich, machine-readable logs with sampling and sensitive-data censoring.
        </p>
        <a class="feature__link" href="{{ '/LOGGING_QUICK_REFERENCE/' | relative_url }}">Logging quick reference</a>
      </div>
    </div>
  </div>
</section>

<section class="section section--muted" id="quickstart">
  <div class="container">
    <header class="section__head">
      <span class="section__eyebrow">Quick start</span>
      <h2 class="section__title">Up and running in a few commands</h2>
      <p class="section__lede">
        Clone the repository, create a virtualenv, install AgentFarm, and run your first simulation.
      </p>
    </header>

    <div class="quickstart">
      <div class="quickstart__steps">
        <div class="quickstart__step">
          <span class="quickstart__step-num">1</span>
          <div>
            <h4>Clone &amp; create a virtualenv</h4>
            <p>AgentFarm targets Python 3.10+. A virtualenv keeps experiment dependencies isolated.</p>
          </div>
        </div>
        <div class="quickstart__step">
          <span class="quickstart__step-num">2</span>
          <div>
            <h4>Install in editable mode</h4>
            <p>Editable installs make it easy to extend agents, environments, or analysis pipelines.</p>
          </div>
        </div>
        <div class="quickstart__step">
          <span class="quickstart__step-num">3</span>
          <div>
            <h4>Run a simulation</h4>
            <p>The CLI emits a SQLite <code>.db</code> under <code>simulations/</code> for downstream analysis.</p>
          </div>
        </div>
        <div class="quickstart__step">
          <span class="quickstart__step-num">4</span>
          <div>
            <h4>Open the docs</h4>
            <p>Browse the <a href="{{ '/README/' | relative_url }}">documentation index</a> for guides, API, and case studies.</p>
          </div>
        </div>
      </div>

<pre class="quickstart__code"><span class="c"># 1. Clone</span>
git clone https://github.com/Dooders/AgentFarm.git
<span class="p">cd</span> AgentFarm

<span class="c"># 2. Virtualenv + install</span>
python -m venv venv
<span class="p">source</span> venv/bin/activate
pip install -r requirements.txt
pip install -e .

<span class="c"># 3. Run a small simulation</span>
python run_simulation.py \
  --environment development \
  --steps 500

<span class="c"># 4. Run the test suite</span>
pytest -q
</pre>
    </div>
  </div>
</section>

<section class="section">
  <div class="container">
    <header class="section__head">
      <span class="section__eyebrow">Latest from the devlog</span>
      <h2 class="section__title">What we've been building</h2>
      <p class="section__lede">
        Build notes, design decisions, and experiment outcomes from the AgentFarm team.
      </p>
    </header>

    <ul class="posts">
      <li>
        <a class="post-card" href="{{ '/devlog/2026-04-23-evolving-hyperparameter-genomes-foraging-learning-agents/' | relative_url }}">
          <span class="post-card__date">April 23, 2026</span>
          <h3 class="post-card__title">Evolving hyperparameter genomes in foraging and learning agents</h3>
          <p class="post-card__excerpt">
            Each agent carries its own hyperparameter chromosome, offspring inherit it (with mutation and
            crossover), and selection is whatever the resource environment happens to apply.
          </p>
          <span class="post-card__more">Read the post</span>
        </a>
      </li>
      <li>
        <a class="post-card" href="{{ '/devlog/2026-04-17-dna-hyperparameter-evolution/' | relative_url }}">
          <span class="post-card__date">April 17, 2026</span>
          <h3 class="post-card__title">DNA-style hyperparameter evolution results</h3>
          <p class="post-card__excerpt">
            Design and initial outcomes of the genetics-inspired hyperparameter evolution work in AgentFarm.
          </p>
          <span class="post-card__more">Read the post</span>
        </a>
      </li>
    </ul>

    <p class="text-center" style="margin-top: 28px;">
      <a class="btn btn--primary" href="{{ '/devlog/' | relative_url }}">Browse the full devlog</a>
    </p>
  </div>
</section>

<section class="cta">
  <h2>Build, evolve, and study agent populations</h2>
  <p>
    Whether you're researching emergent behavior, comparing learning algorithms, or running parameter
    sweeps, AgentFarm gives you the primitives and the data to do it reproducibly.
  </p>
  <div class="hero__cta" style="justify-content: center;">
    <a class="btn btn--primary" href="{{ '/README/' | relative_url }}">Get started</a>
    <a class="btn btn--ghost" href="https://github.com/Dooders/AgentFarm" target="_blank" rel="noopener">Star on GitHub ↗</a>
  </div>
</section>
