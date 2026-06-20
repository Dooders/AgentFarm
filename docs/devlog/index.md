---
layout: page
title: Devlog
subtitle: Build notes, design decisions, and experiment outcomes from AgentFarm development.
---

<ul class="posts">
  <li>
    <a class="post-card" href="{{ '/devlog/2026-06-20-transferable-signal-budget/' | relative_url }}">
      <span class="post-card__date">2026-06-20</span>
      <h3 class="post-card__title">The transferable-signal gate: do learned policies beat their own init?</h3>
      <p class="post-card__excerpt">
        The opening step of the #904 inheritance-ladder experiment: before
        building richer P2-P4 payloads, confirm there is anything worth
        inheriting. In a learning-positive regime (8 agents, 3000 steps,
        reproduction disabled), paired held-out rollouts under the
        non-degenerate weighted policy show a modest but robust early-age
        decision-quality signal in all three profiles (~+15–30 net reward,
        95% CIs exclude zero). The gate passes at a realistic effect size,
        justifying the richer-payload work.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="https://github.com/Dooders/AgentFarm/issues/904">Inheritance-ladder experiment (#904)</a>,
      <a href="{{ '/design/inherited_payload_design/' | relative_url }}">Inherited payload design</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-06-09-every-agent-a-different-goal/' | relative_url }}">
      <span class="post-card__date">2026-06-09</span>
      <h3 class="post-card__title">When every agent has a different goal</h3>
      <p class="post-card__excerpt">
        Making the reward function itself a per-agent, heritable trait. Across
        20 paired seeds, a population where each agent optimizes a different
        randomly-drawn objective carries ~40% fewer agents than the matched
        hand-tuned control and collapses its behavior toward gathering
        (+16.9pp). The goal diversity persists for the whole run, and every
        effect is huge and significant — un-curated objective diversity lowers
        collective fitness.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/experiments/intrinsic_evolution/intrinsic_goals/' | relative_url }}">Intrinsic goals experiment doc</a>,
      <a href="{{ '/design/hyperparameter_chromosome/' | relative_url }}">Hyperparameter chromosome design</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-06-04-are-we-measuring-at-the-wrong-level/' | relative_url }}">
      <span class="post-card__date">2026-06-04</span>
      <h3 class="post-card__title">Are we measuring at the wrong level?</h3>
      <p class="post-card__excerpt">
        Re-scoring the 36-run inheritance A/B at the newborn level. Warm-start
        produces two small, robust behavioral shifts — slightly fewer negative
        actions, but slightly lower net RL reward — and neither is a fitness
        gain; survival and resources don't move. The population-level null
        wasn't a measurement artifact.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/devlog/2026-05-21-baldwinian-vs-lamarckian-ab-harness/' | relative_url }}">The inheritance A/B this follows up</a>,
      <a href="{{ '/experiments/intrinsic_evolution/inheritance_mode_ab/' | relative_url }}">Inheritance A/B experiment doc</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-05-21-baldwinian-vs-lamarckian-ab-harness/' | relative_url }}">
      <span class="post-card__date">2026-05-21</span>
      <h3 class="post-card__title">Baldwinian vs Lamarckian: policy warm-start across three resource regimes</h3>
      <p class="post-card__excerpt">
        Full 36-run matched matrix (2 arms × 3 profiles × 6 seeds). Lamarckian
        warm-start applied ~85% of the time and paired runs diverged, but no
        profile cleared the robustness gate — keep Baldwinian as default for now.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/experiments/intrinsic_evolution/inheritance_mode_ab/' | relative_url }}">Inheritance A/B experiment doc</a>,
      <a href="{{ '/devlog/2026-04-23-evolving-hyperparameter-genomes-foraging-learning-agents/' | relative_url }}">Original Baldwinian context</a>.
    </p>
  </li>
  <li>
    <a class="post-card" href="{{ '/devlog/2026-05-18-gene-flow-and-the-buffer/' | relative_url }}">
      <span class="post-card__date">2026-05-18</span>
      <h3 class="post-card__title">Gene flow and the buffer</h3>
      <p class="post-card__excerpt">
        A crossover-enabled rerun closes the buffer arc with a
        profile-dependent result: conservative speciation compresses
        under gene flow, buffered trajectories still diverge, and
        balanced stays noisy.
      </p>
      <span class="post-card__more">Read the post</span>
    </a>
    <p class="post-card__excerpt">
      Related docs:
      <a href="{{ '/devlog/2026-05-12-seed-sweep-reality-check/' | relative_url }}">Replication baseline</a>,
      <a href="{{ '/experiments/intrinsic_evolution/crossover_rerun/' | relative_url }}">Crossover rerun experiment doc</a>.
    </p>
  </li>
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
