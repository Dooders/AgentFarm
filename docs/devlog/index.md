---
layout: page
title: Devlog
subtitle: Build notes, design decisions, and experiment outcomes from AgentFarm development.
---

<ul class="posts">
  <li>
    <a class="post-card" href="{{ '/devlog/2026-05-16-is-the-dqn-actually-learning/' | relative_url }}">
      <span class="post-card__date">2026-05-16</span>
      <h3 class="post-card__title">Is the DQN actually learning?</h3>
      <p class="post-card__excerpt">
        A user suspected agents weren't learning. Instrumenting the
        decision module surfaced four real bugs in one stack — a global
        training throttle, a never-applied epsilon schedule, a
        YAML-to-config mapping that dropped knobs on the floor, and a
        hidden-size field that did nothing. After fixing all four,
        training volume jumps ~9× and lifespan +23%, but the late-vs-
        early decision-quality signal is still small. The remaining
        bottleneck is the simulation's signal-to-noise ratio, not the
        code.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/deep_q_learning/' | relative_url }}">Deep Q-learning module reference</a>,
      <a href="{{ '/design/hyperparameter_chromosome/' | relative_url }}">Hyperparameter chromosome design</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-05-12-seed-sweep-reality-check/' | relative_url }}">
      <span class="post-card__date">2026-05-12</span>
      <h3 class="post-card__title">When one seed disagrees with six</h3>
      <p class="post-card__excerpt">
        A 6-seed-per-profile follow-up to the resource-buffer comparison.
        Speciation always diverges; the learning_rate and ensemble_size
        "flips" were single-seed artifacts; a couple of gene-level
        patterns survive but only as magnitude trends.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/devlog/2026-05-04-resource-buffer-shapes-intrinsic-evolution/' | relative_url }}">Prior devlog</a>,
      <a href="{{ '/experiments/intrinsic_evolution/intrinsic_evolution/' | relative_url }}">Intrinsic evolution docs</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-05-04-resource-buffer-shapes-intrinsic-evolution/' | relative_url }}">
      <span class="post-card__date">2026-05-04</span>
      <h3 class="post-card__title">Does the resource buffer pick the genes?</h3>
      <p class="post-card__excerpt">
        Three intrinsic-evolution runs share every policy and only differ in their stable
        resource profile. Most behavioural genes drift the same way, but learning rate,
        ensemble size, and the speciation trajectory split cleanly along the buffer.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/glossary/' | relative_url }}">Glossary</a>,
      <a href="{{ '/experiments/intrinsic_evolution/intrinsic_evolution/' | relative_url }}">Intrinsic evolution docs</a>,
      <a href="{{ '/design/hyperparameter_chromosome/' | relative_url }}">Hyperparameter chromosome design</a>,
      <a href="{{ '/devlog/2026-04-23-evolving-hyperparameter-genomes-foraging-learning-agents/' | relative_url }}">Companion devlog</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-04-23-evolving-hyperparameter-genomes-foraging-learning-agents/' | relative_url }}">
      <span class="post-card__date">2026-04-23</span>
      <h3 class="post-card__title">Evolving hyperparameter genomes in foraging and learning agents</h3>
      <p class="post-card__excerpt">
        How much adaptive behavior can emerge from ecology alone — finite resources, costly reproduction,
        and inherited learning priors — without hand-crafted fitness functions? A small step toward
        answering that, with chromosomes attached to each agent.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/glossary/' | relative_url }}">Glossary</a>,
      <a href="{{ '/design/hyperparameter_chromosome/' | relative_url }}">Hyperparameter chromosome design</a>,
      <a href="{{ '/experiments/intrinsic_evolution/intrinsic_evolution/' | relative_url }}">Intrinsic evolution docs</a>,
      <a href="{{ '/devlog/2026-05-04-resource-buffer-shapes-intrinsic-evolution/' | relative_url }}">Follow-up devlog</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-04-17-dna-hyperparameter-evolution/' | relative_url }}">
      <span class="post-card__date">2026-04-17</span>
      <h3 class="post-card__title">DNA-style hyperparameter evolution results</h3>
      <p class="post-card__excerpt">
        Design and initial outcomes of the genetics-inspired hyperparameter evolution work in AgentFarm,
        including the typed gene representation and reproduction-time evolution wiring.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
  </li>
</ul>
