# AgentFarm Twitter/X account plan

Launch plan for the AgentFarm account: profile assets, voice, and a 3-week
schedule of three tweets per day covering the project, its intentions, design
principles, and research results.

All content below is grounded in the repository and docs site
([dooders.github.io/AgentFarm](https://dooders.github.io/AgentFarm/)).
Adjust dates, links, and screenshots at posting time.

---

## Profile

### Handle suggestions

- `@AgentFarmSim` (preferred — "AgentFarm" alone is likely taken)
- `@AgentFarmDev`
- `@FarmOfAgents`

### Display name

**AgentFarm**

### Bio (160-character limit)

Primary:

> Open-source agent-based simulation & RL research platform. Digital ecology: evolving genomes, learning agents, honest results — negative ones too. 🧪 Apache-2.0

Alternates:

> Simulating digital ecologies: learning agents, heritable genomes, emergent dynamics. Open-source research platform in Python. Devlog + code below. 🌱

> We grow agents. Multi-agent sims, DQN decision stacks, evolvable hyperparameters, and a devlog that publishes the null results too. Python · Apache-2.0

### Location / website fields

- Location: `simulations/*.db`
- Website: `https://dooders.github.io/AgentFarm/`

### Pinned tweet

Use Day 1, Tweet 1 (the launch thread opener) as the pinned tweet.

---

## Profile picture prompt

For an image-generation model, square output (400×400 minimum, generate at 1024×1024):

> Minimal flat vector logo mark for a software project called "AgentFarm".
> A stylized square field-grid seen from above, like farmland plots crossed
> with a simulation grid. A few of the grid cells contain small glowing dots
> (agents) in warm amber, one cell sprouts a tiny abstract seedling made of
> circuit-like lines. Color palette: deep forest green and dark teal
> background, amber/gold accent dots, thin off-white grid lines. Geometric,
> modern, high contrast, no text, no gradients heavier than subtle, reads
> clearly at 48×48 pixels. Centered emblem with generous margin.

## Banner image prompt

For an image-generation model, 3:1 ratio (X banner is 1500×500):

> Wide panoramic illustration of a digital ecosystem simulation, flat vector
> style with a dark scientific-dashboard aesthetic. A top-down 2D grid world
> stretches across the banner: clusters of glowing amber resource nodes,
> dozens of small agent dots in teal and green with faint motion trails,
> sparse territory boundaries. On the right third, subtle translucent
> overlays of line charts and population curves rising and falling, as if the
> world is being measured while it runs. Deep navy/charcoal background, thin
> grid lines, restrained color (teal, green, amber). Leave the left ~25%
> visually quiet so the profile picture and name overlay cleanly. No text.

---

## Voice and style rules

- **Honest instrument, not hype machine.** We publish negative results in the
  devlog; the account does the same. Never oversell.
- Plain language first, jargon second. Define terms inline when cheap.
- Every claim about results links to a devlog post or experiment doc.
- Threads for research stories; single tweets for tips and principles.
- Hashtags: at most two per tweet, drawn from `#ALife`, `#ReinforcementLearning`,
  `#AgentBasedModeling`, `#OpenSource`, `#ComplexSystems`, `#Python`.
- Media: screenshots of charts from `farm/analysis` output, devlog figures
  (`docs/research/devlog/figures/`), and short sim GIFs beat text-only tweets.

Suggested posting slots (adjust to audience analytics): **9:00**, **13:00**,
**18:00** local time.

---

## Three-week schedule (3 tweets/day)

Legend: 🌅 morning slot · ☀️ midday slot · 🌆 evening slot.
`[link: …]` marks where to attach a URL; `[media: …]` marks a suggested image.

### Week 1 — What AgentFarm is

**Day 1 (Mon) — Launch**

- 🌅 Introducing AgentFarm 🌱 — an open-source platform for growing digital
  ecologies: multi-agent simulations where agents perceive, learn, reproduce,
  and pass on heritable genomes. Built in Python, measured in SQLite, published
  honestly. Follow along. [link: repo] [media: banner art or sim GIF]
- ☀️ Why "farm"? Because we don't script behaviors — we plant agents in a
  resource-constrained world and see what grows. Selection pressure comes from
  the ecology itself: finite food, costly reproduction, real death.
- 🌆 What you'll get from this account: build notes, design principles,
  experiment results (including the failures), and ways to run it all
  yourself. Everything is Apache-2.0. [link: repo]

**Day 2 (Tue) — Getting started**

- 🌅 From zero to a living world in four commands:
  clone → venv → `pip install -r requirements.txt` → `python run_simulation.py
  --environment development --steps 1000`.
  Your first ecosystem lands in `simulations/*.db`. [link: quick start docs]
- ☀️ Every AgentFarm step runs the same pipeline: refresh perception channels →
  agents act → world resolves combat/sharing/reproduction → observations
  update → everything persists to SQLite → the dead are cleaned up. Simple
  loop, complex outcomes.
- 🌆 A simulation you can't interrogate afterwards is just a screensaver.
  Every run writes agent states, actions, reproduction events, and learning
  telemetry to a queryable database. The sim is the experiment; the DB is the
  lab notebook.

**Day 3 (Wed) — Agents**

- 🌅 An AgentFarm agent is not a monolith. It's an AgentCore plus pluggable
  components — movement, perception, combat, learning — and a swappable
  behavior policy. Want a new species? Compose one; don't fork the engine.
- ☀️ Agents see the world through egocentric multi-channel tensors: allies,
  enemies, resources, visibility, trails, damage heat — each a layer in a
  (channels × H × W) window centered on the agent. Just like a tiny
  convolutional retina. [media: observation channel figure]
- 🌆 Actions are a registry, not an enum: move, gather, attack, share,
  reproduce, pass — and yours. Register a new action at import time and every
  agent's action space grows to fit. No core edits required.

**Day 4 (Thu) — The world**

- 🌅 The world is a 2D grid with regenerating resources and a spatial index
  underneath — KD-tree, quadtree, or spatial hash, your pick. Proximity
  queries with dirty-region tracking mean only changed cells get reindexed.
- ☀️ Determinism is a feature, not an accident. `run_simulation.py` pins
  `PYTHONHASHSEED`, seeds are explicit, and identical configs replay identical
  worlds. If you can't reproduce it, you can't study it. [link: deterministic
  simulations guide]
- 🌆 Ecology is the fitness function. No hand-written "score" tells agents how
  to live — energy budgets, resource scarcity, and reproduction costs do the
  selecting. What survives is what worked.

**Day 5 (Fri) — Learning & evolution**

- 🌅 Each agent can carry a DQN decision stack — and its hyperparameters
  (learning rate, epsilon, network width…) live on a typed chromosome that
  offspring inherit, cross over, and mutate. Learning parameters become
  heritable traits. [link: hyperparameter chromosome design]
- ☀️ Two timescales of adaptation in one sim: within a lifetime, agents learn
  by RL; across generations, genomes evolve by selection. Watching them
  interact is the whole research program.
- 🌆 We call it intrinsic evolution: no external fitness function, no
  generational reset, no evaluation phase. Agents that eat, survive, and
  reproduce pass on their genes mid-simulation. Selection just… happens.
  [link: intrinsic evolution experiment]

**Day 6 (Sat) — Data & analysis**

- 🌅 The analysis stack mirrors a real lab: SimulationDatabase → repositories →
  analyzers → reports and plots. Population, spatial, combat, and learning
  modules ship in the box. [media: analysis chart]
- ☀️ Charts we stare at daily: population curves, resource-vs-agent phase
  plots, per-generation gene distributions, reward trajectories by age. All
  generated from the run DB, all reproducible from the seed.
- 🌆 Weekend idea: run 1,000 steps, open the `.db` in any SQLite browser, and
  ask your own questions. The schema covers agent states, actions, and
  reproduction events. Your queries are as first-class as ours.

**Day 7 (Sun) — Recap & docs**

- 🌅 Week 1 recap: AgentFarm = composable agents + ecological selection +
  learning under pressure + everything persisted for analysis. Open-source,
  Python-first, Apache-2.0. [link: repo]
- ☀️ The docs site has role-based entry points: researchers start at the
  experiments catalog, developers at architecture, operators at deployment.
  [link: docs site]
- 🌆 Question for the timeline: what emergent behavior would convince YOU a
  simulated ecology is doing something real? We collect answers — some become
  experiments.

### Week 2 — Intentions & design principles

**Day 8 (Mon) — Why we build this way**

- 🌅 Intention #1: AgentFarm is an *instrument*, not a demo. Instruments must
  be calibrated, reproducible, and honest about their noise floor. That
  drives every design decision — from pinned seeds to typed configs.
- ☀️ Intention #2: negative results are results. Our devlog publishes the
  experiments that failed, the "effects" that were single-seed artifacts, and
  the nulls that held up. Science rots without them. [link: devlog]
- 🌆 Intention #3: extensible by strangers. If adding a new observation
  channel, action, or behavior requires editing engine internals, we've
  failed. Registries and interfaces exist so your fork stays small.

**Day 9 (Tue) — Composition over inheritance**

- 🌅 Design principle: composition over inheritance. Agents gain abilities by
  adding components (movement, perception, combat, learning), not by
  descending from a god-class. Deep hierarchies fossilize; components stay
  swappable.
- ☀️ Concrete payoff: a memory-augmented agent is AgentCore + a memory
  component + the same behavior interface. No parallel class tree, no
  diamond problem, no re-testing the whole ancestry.
- 🌆 Single Responsibility in practice: the environment orchestrates, the
  resource manager owns resources, the spatial index owns proximity, the
  metrics tracker observes. When one thing changes, one module changes.

**Day 10 (Wed) — Open for extension**

- 🌅 Open–Closed Principle, applied: new actions and observation channels
  register themselves at import time. The engine never learns their names.
  Adding a "pheromone" channel touches zero core files.
- ☀️ Dependency Inversion, applied: the core depends on `DatabaseProtocol` and
  `RepositoryProtocol`, not SQLite. Tests inject in-memory fakes; production
  wires the real thing. Swappable persistence for free. [link: DI docs]
- 🌆 The test suite is ~6,500 tests and counting. Protocol-based seams are why
  that's tractable — most tests never touch a real database, filesystem, or
  network.

**Day 11 (Thu) — Simplicity & configuration**

- 🌅 KISS survives contact with research code only if you defend it. Our rule:
  the simulation loop reads like the six-line pseudocode in the docs. All
  cleverness lives behind interfaces where it can be replaced.
- ☀️ Every run is a `SimulationConfig`: typed, nested, YAML-loaded, and
  validated before step one. Population, resources, learning, observation —
  all declared, all diffable, all reproducible. Configs are experiments.
- 🌆 DRY, but not prematurely: sweeps don't copy configs, they declare
  variations. The ExperimentRunner takes a base config + a list of overrides
  and fans out the runs. [link: experiment runner guide]

**Day 12 (Fri) — Reproducibility**

- 🌅 Hard-won lesson: one seed is an anecdote. Our CohortRunner wraps any
  experiment in N independent seeded runs and aggregates mean, deviation, and
  convergence stats. Single-run "discoveries" rarely survive it.
- ☀️ True story: an early result showed learning-rate "flips" between resource
  regimes. Exciting! A 6-seed sweep later: single-seed artifact. Gone. The
  devlog post stays up because that's the point. [link: seed-sweep devlog]
- 🌆 Reproducibility stack: pinned `PYTHONHASHSEED` → explicit RNG seeds →
  deterministic spatial updates → versioned configs → archived run DBs. Any
  figure in our devlog can be regenerated from a command line.

**Day 13 (Sat) — Performance**

- 🌅 Simulation perf principle: never recompute what didn't change.
  Dirty-region tracking on the spatial index means a mostly-idle world
  reindexes almost nothing per step.
- ☀️ Observations can be sparse or dense, CPU or GPU, at your chosen dtype —
  because a thousand agents each holding a multi-channel float tensor adds up
  fast. `ObservationConfig` makes the trade-off explicit.
- 🌆 We benchmark the spatial backends against each other (KD-tree vs quadtree
  vs hash) with committed, verified benchmark artifacts — so perf claims in
  the docs are checkable, like everything else. [link: benchmarks]

**Day 14 (Sun) — Engineering culture recap**

- 🌅 Week 2 recap — the principles: composition over inheritance ·
  open–closed registries · protocol-based DI · typed config as experiment ·
  multi-seed or it didn't happen · benchmarked perf claims.
- ☀️ Structured logging everywhere (`structlog`): every event carries context —
  step, agent, values — machine-parseable and human-readable. When a sim
  misbehaves at step 8,412, grep is a research tool.
- 🌆 None of this is exotic. It's boring discipline applied to research code,
  where discipline is rarest and pays most. Steal any of it for your own
  projects — that's what the license is for.

### Week 3 — Research stories & community

**Day 15 (Mon) — Intrinsic evolution arc**

- 🌅 Research thread week 🧵 Day 1: we gave every agent its own heritable
  hyperparameter genome and removed the fitness function entirely. Survival
  and reproduction in a shared resource world do all the selecting.
  [link: intrinsic evolution devlog]
- ☀️ Result: gene distributions drift, split, and stabilize across a
  10,000-step run — no evaluator, no generations, no reset. The ecology alone
  produces evolutionary dynamics. [media: gene trajectory figure]
- 🌆 The question behind it: how much adaptive behavior can emerge from
  ecology alone — finite resources, costly reproduction, inherited priors —
  without a human writing "fitness"? That's AgentFarm's north star.

**Day 16 (Tue) — The environment picks the genes**

- 🌅 Experiment: three identical populations, one knob changed — the stability
  of the resource supply. Most behavioral genes drift the same way regardless.
  But learning rate and ensemble size split cleanly along the resource buffer.
  [link: resource-buffer devlog]
- ☀️ Interpretation: the environment doesn't just select behaviors — it
  selects *learning styles*. Stable worlds and volatile worlds favor
  measurably different learners.
- 🌆 Then we ran the humility check: 6 seeds per profile. The headline
  gene "flips" didn't replicate; the divergence patterns did, as magnitude
  trends. Both facts are in the devlog. [link: seed-sweep devlog]

**Day 17 (Wed) — Is the DQN actually learning?**

- 🌅 A user asked: "are your agents actually learning, or just wandering?"
  Fair. We instrumented the decision stack and found FOUR real bugs — a
  global training throttle, a never-applied epsilon schedule, a config field
  that dropped knobs, a hidden-size that did nothing. [link: DQN devlog]
- ☀️ After the fixes: ~9× more training updates, +23% agent lifespan. But
  the late-life vs early-life decision-quality gap stayed small. The
  bottleneck moved from our code to the environment's signal-to-noise. Bugs
  fixed ≠ question answered.
- 🌆 The meta-lesson: in RL + simulation, "it runs" and "it learns" are
  separated by instrumentation. Log the training volume. Plot decision
  quality by age. Assume nothing.

**Day 18 (Thu) — Baldwinian vs Lamarckian**

- 🌅 Should offspring inherit their parents' learned neural weights
  (Lamarckian) or just the genome, learning from scratch (Baldwinian)?
  We built a paired A/B harness — 36 matched runs, 3 resource regimes — to
  find out. [link: inheritance A/B devlog]
- ☀️ Result: Lamarckian warm-start applied ~85% of the time, runs genuinely
  diverged… and no regime cleared our robustness gate. Baldwinian stays the
  default. A publishable null, published.
- 🌆 We then re-scored the same runs at the newborn level to check we weren't
  measuring at the wrong altitude. Two small behavioral shifts, no fitness
  gain. The null held. [link: measurement-level devlog]

**Day 19 (Fri) — When every agent wants something different**

- 🌅 What if the reward function itself were a heritable, per-agent trait?
  We ran 20 paired seeds where each agent optimizes its own randomly-drawn
  objective vs a hand-tuned control. [link: goal-diversity devlog]
- ☀️ Result: goal-diverse populations carried ~40% fewer agents and collapsed
  behavior toward gathering (+17pp). Objective diversity persisted all run —
  and un-curated diversity *lowered* collective fitness. Big effects, all
  significant.
- 🌆 Why it matters beyond ALife: multi-agent systems where each unit
  optimizes a private objective are everywhere (markets, swarms, orgs).
  Here's a controlled world where you can watch that go wrong.

**Day 20 (Sat) — The inheritance ladder**

- 🌅 The big one: a 90-run ladder testing five inheritance payloads, from bare
  genome up to full learned-policy transfer. Graded on offspring early-life
  reward. Hypothesis: richer inheritance → better start. [link: inheritance
  ladder devlog]
- ☀️ Result: every robust effect was a LOSS. The richer the inherited payload,
  the worse the offspring did — warm-started agents clamped to low,
  ecology-blind trajectories while cold-start offspring tracked the world.
  We re-ran under low density to kill the confound. It held.
- 🌆 Three devlog posts, ~200 runs, and the honest conclusion: for this
  design, inherited policies hurt. That's not failure — that's the instrument
  working. Now we get to ask *why*, with data.

**Day 21 (Sun) — What's next & how to join**

- 🌅 Week 3 recap: ecology-only evolution works · environments select learning
  styles · four DQN bugs found & fixed · inheritance nulls that held up ·
  goal diversity that backfired. All reproducible from the repo. [link:
  devlog index]
- ☀️ Where AgentFarm goes next: richer inherited payloads, memory-augmented
  agents, better decision-quality metrics, and more multi-seed rigor.
  The experiments catalog is the living roadmap. [link: experiments catalog]
- 🌆 Want in? Run a sim, open an issue, replicate a devlog result, or pick up
  a starter task. Research code that strangers can extend is the whole
  point — come extend it. 🌱 [link: CONTRIBUTING.md]

---

## Automating the posts

The schedule is duplicated in machine-readable form at
[`tweet_schedule.json`](tweet_schedule.json), and
[`scripts/post_scheduled_tweet.py`](../../scripts/post_scheduled_tweet.py)
posts whichever tweet is due at the current UTC time. The intended trigger is
a **Cursor Automation** with a cron schedule.

### 1. Get X API access

As of 2026 there is **no free tier** for new X developers — the API is
pay-per-use (roughly $0.015 per plain post, $0.20 per post containing a URL;
legacy Basic/Pro tiers are closed to new signups). For this campaign
(63 tweets, ~24 with links) expect on the order of **$5–6 total**.

1. Sign up at [developer.x.com](https://developer.x.com) with the
   `@AgentFarm…` account and add a small credit balance.
2. Create a Project + App. Under **User authentication settings**, enable
   **Read and Write** permissions (OAuth 1.0a).
3. From the app's **Keys and tokens** page, collect four values:
   API Key, API Key Secret, Access Token, Access Token Secret
   (the access token must be generated *after* enabling write permissions,
   and must belong to the AgentFarm account).

### 2. Store the credentials as Cursor secrets

In [cursor.com/dashboard/cloud-agents](https://cursor.com/dashboard/cloud-agents)
→ **Secrets**, add four **Runtime Secrets** (runtime secrets are injected as
environment variables but redacted from transcripts and tool results):

| Secret name | Value |
|-------------|-------|
| `X_API_KEY` | API Key |
| `X_API_SECRET` | API Key Secret |
| `X_ACCESS_TOKEN` | Access Token |
| `X_ACCESS_TOKEN_SECRET` | Access Token Secret |

Scope them to this repository so only AgentFarm agents receive them.

### 3. Create the Cursor Automation

At [cursor.com/automations/new](https://cursor.com/automations/new)
(or via the `/automate` skill in a local agent session):

- **Trigger:** Scheduled, cron `0 9,13,18 * * *` (UTC) — one firing per slot.
- **Repository:** `Dooders/AgentFarm`, branch `main` (a repository must be
  selected; cron automations default to "no repository", which would hide
  the script and secrets scoped to this repo).
- **Prompt:**

  > Run `python scripts/post_scheduled_tweet.py` from the repository root.
  > If it prints "No tweet due", stop — that is expected outside campaign
  > slots. If it exits non-zero for any other reason, report the full error
  > output. Do not edit any files, do not open a PR, and do not retry a
  > successful post.

- **Tools:** disable *Pull request creation* — this automation only posts.

The script is stateless by design: each cron firing posts exactly the tweet
whose slot covers the current time (a 2-hour lateness window per slot, and
slots are ≥4 hours apart, so a run can never double-post). If a firing is
missed entirely, that slot's tweet is skipped rather than posted late.

### Verifying before launch

```bash
# Show the whole expanded schedule
python scripts/post_scheduled_tweet.py --list

# Preview what a given slot would post, without credentials
python scripts/post_scheduled_tweet.py --dry-run --at 2026-07-20T09:00:00

# One real end-to-end post (uses the four X_* env vars)
python scripts/post_scheduled_tweet.py --at 2026-07-20T09:00:00
```

To shift the campaign, edit `campaign.start_date` (and `slot_hours` for
different posting times) in `tweet_schedule.json` — the script derives every
slot from those values.

## After week 3

Sustainable cadence once the launch backlog is exhausted:

- **1/day minimum:** alternate devlog announcements, design notes, and
  figures from ongoing runs.
- **Per devlog post:** a 3–5 tweet thread summarizing question → method →
  result → link.
- **Monthly:** a recap thread and a "replicate this figure" challenge with
  the exact command line.
