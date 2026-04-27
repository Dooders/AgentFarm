# Glossary

## Agent

An agent is an autonomous entity that perceives its environment, makes decisions, and takes actions toward survival or other objectives.

In AgentFarm, an agent is modeled as a **stateful simulation actor with configurable behavior, resources, and lifecycle dynamics**:

- Agents hold internal state (for example energy, generation, and ancestry metadata) and interact with the environment through action systems.
- Agent behavior can be driven by configured modules and learning-related parameters, which may be inherited and transformed across generations.
- Agent interactions with resources, space, and other agents generate the population-level patterns analyzed throughout the library.

In this library, agents are the core units of simulation from which lineage, ecological dynamics, and emergent behavior are derived.

## Agency

Agency is the capacity of an agent to select and execute actions based on its internal state, available information, and goals.

In AgentFarm, agency is modeled as **state-dependent decision-making under environment and resource constraints**:

- Agents evaluate their current context (for example energy level, local conditions, and policy state) to choose among available actions.
- Agency is bounded rather than absolute: feasible choices are shaped by simulation rules, action availability, and lifecycle constraints.
- Differences in agency expression (how consistently and effectively agents choose actions) influence both learning outcomes and downstream evolutionary pressure.

In this library, agency is the functional decision capacity that turns an agent from a passive entity into an adaptive actor.

## Action

An action is a discrete choice or operation an agent executes to affect its state or its environment.

In AgentFarm, action is modeled as an **agent-level decision step evaluated against simulation rules and resource constraints**:

- At each update step, agents select from available behaviors (for example movement, interaction, or resource-related operations) based on their current policy and state.
- Actions consume or transform state variables (such as energy) and can change local environment conditions and future decision context.
- Action outcomes provide the immediate feedback that drives learning dynamics within an agent's lifetime.

In this library, actions are the operational link between decision policies and measurable simulation outcomes.

## Environment

An environment is the external world context in which agents exist, act, and experience constraints or opportunities.

In AgentFarm, environment is modeled as a **simulation state and rule system that defines conditions, resources, and interaction dynamics**:

- The environment provides the spatial and resource context agents operate within (for example positions, available resources, and local competition).
- It defines constraints and update rules that shape outcomes such as survival, movement, and reproduction eligibility.
- Environment-level dynamics create the selective conditions under which inherited strategies are advantaged or disadvantaged over time.

In this library, environment is the structured simulation context that turns agent behavior into measurable ecological and evolutionary dynamics.

## Ecology

Ecology is the study of how organisms interact with each other and with their environment, and how those interactions shape population-level patterns.

In AgentFarm, ecology is modeled as **emergent interaction structure across agents, resources, and environmental constraints**:

- Ecological dynamics arise from repeated local interactions such as competition, resource access, movement, and reproduction.
- These interactions produce measurable population-level effects (for example distribution shifts, energy patterns, and differential persistence across groups).
- Ecology links micro-level behavior (individual actions and decisions) to macro-level outcomes that later feed niche formation, selection, and speciation analyses.

In this library, ecology is the interaction layer that connects agent behavior and environment rules to emergent evolutionary context.

## Energy

Energy is a quantitative resource reserve that supports activity, maintenance, and survival.

In AgentFarm, energy is modeled as an **agent state variable that mediates behavior and lifecycle outcomes**:

- Agents spend energy while acting and can gain energy through successful resource interactions.
- Energy level influences survival pressure and can constrain whether agents persist long enough to reproduce.
- Energy-related telemetry (for example `mean_energy`) is used in niche and population analyses to connect behavioral/ecological context to evolutionary outcomes.

In this library, energy is both a local control signal for individual agents and a global driver of population-level dynamics.

## Learning

Learning is the process by which an agent updates its behavior from experience to improve decisions under changing conditions.

In AgentFarm, learning is modeled as **experience-driven policy adaptation governed by configurable learning parameters**:

- Agents can adjust action tendencies over time based on environment interaction outcomes (for example reward-like or success/failure signals).
- Learning behavior is controlled by evolvable hyperparameters (for example `learning_rate`, `gamma`, and `epsilon_decay`) represented in chromosomes.
- This creates a bridge between within-lifetime adaptation (learning dynamics) and across-generation adaptation (selection on inherited learning settings).

In this library, learning is the mechanism that turns interaction history into behavioral change, and it is a key substrate for evolutionary pressure on agent strategies.

## Gene

A gene is a unit of inherited information that can vary across individuals and influence traits.

In AgentFarm, a gene is modeled as a **typed hyperparameter locus** inside a `HyperparameterChromosome`:

- Each gene has a name and numeric value (for example `learning_rate`, `gamma`, or `epsilon_decay`), along with bounds and mutation behavior.
- Genes can be marked evolvable or fixed; mutation operators act on evolvable genes.
- Per-gene population statistics (mean, median, variance-related measures, boundary fractions) are used to track adaptation over simulation time.

In this library, a gene is not DNA; it is an evolvable control parameter that shapes agent learning and behavior.

## Chromosome

A chromosome is an organized collection of genes inherited and transformed across generations.

In AgentFarm, a chromosome is modeled as a **structured bundle of evolvable agent parameters** (currently represented by `HyperparameterChromosome`):

- A chromosome contains multiple genes with values, bounds, and mutation/crossover semantics.
- During reproduction, offspring chromosomes are derived by inheritance plus optional crossover and mutation, depending on the active reproduction policy.
- Chromosomes are the primary unit used for trajectory logging, clustering/speciation analysis, and lineage-linked evolutionary tracking.

In practice, a chromosome here is an evolvable parameter configuration for an agent, not a biological genome.

## Genome

A genome is the complete set of inherited information that defines an individual.

In AgentFarm, a genome is modeled as a **serializable agent-level configuration and ancestry representation**:

- The `farm.core.genome` module captures and reconstructs agent state/configuration through a genome dictionary interface.
- `genome_id` fields are used across storage and analysis layers to encode ancestry relationships and lineage grouping.
- Genome-level operations (such as mutation and crossover in the genome utilities) support evolutionary experimentation on inherited agent traits.

In this library, genome refers to computationally encoded inherited agent information, not a biochemical DNA sequence.

## Reproduction

Reproduction is the process by which new individuals are created from existing parent individuals.

In AgentFarm, reproduction is modeled as **offspring creation with inherited and optionally transformed parameters**:

- Reproduction uses parent information to initialize child state and inherited representations (for example chromosome/genome-linked fields).
- Depending on configuration and eligibility, offspring generation can include direct inheritance, crossover with a co-parent, and mutation.
- Reproduction outcomes feed lineage tracking through ancestry metadata and can change population structure over time.

In this library, reproduction is the generational transition mechanism that connects survival dynamics to evolutionary change.

## Inheritance

Inheritance is the transfer of information or traits from parent(s) to offspring.

In AgentFarm, inheritance is modeled as **offspring initialization from parent-derived genome/chromosome information**:

- Reproduction paths pass parent-derived values into child configuration and learning parameters.
- Parent relationships are tracked through lineage metadata (for example `parent_ids` and `genome_id` conventions).
- Inheritance can be direct (copy-like) or combined with crossover and mutation, depending on the active reproduction and evolution settings.

In this library, inheritance is the mechanism that preserves continuity across generations in simulated populations.

## Crossover

Crossover is the recombination of inherited information from multiple parents to form offspring.

In AgentFarm, crossover is modeled as **operator-driven recombination of parent chromosome/genome data**:

- Hyperparameter chromosome utilities provide multiple crossover modes (for example single-point, uniform, blend, and multi-point).
- Crossover can be enabled or disabled by policy/config and is applied during reproduction when eligible co-parents are available.
- The resulting child representation is then optionally followed by mutation, forming a full recombination-plus-variation step.

In this library, crossover increases diversity by mixing existing inherited structures rather than creating variation from scratch.

## Mutation

Mutation is a random or stochastic change to inherited information that introduces variation.

In AgentFarm, mutation is modeled as **parameter perturbation during genetic operations**:

- Hyperparameter chromosomes support mutation operators and modes (for example Gaussian and multiplicative behavior in the chromosome utilities).
- Genome utilities also support mutation of inherited configuration structures.
- Mutation rates and scales control exploration versus stability during evolutionary runs.

In practice, mutation is the primary source of novel variation in the library's inheritance pipeline.

## Selection

Selection is the process by which some inherited variants become more common because they lead to better survival or reproductive outcomes under given conditions.

In AgentFarm, selection is modeled as an **artificially defined selection regime intended to approximate natural-selection-like pressure**:

- The simulation specifies explicit environment rules, constraints, and reward/cost structures that determine which agents persist and reproduce.
- Differential survival and reproduction emerge from those rules, so fitter chromosome/genome configurations contribute more descendants over time.
- This is artificial selection because the pressure is designed by the modeler, but it is natural-selection-like because it operates through competition, resource limits, and inheritance in the simulated environment.

In this library, selection means model-driven evolutionary pressure that produces natural-selection-style population change in silico.

## Lineage

Lineage is the chain of descent connecting an individual or group to its ancestors over time.

In AgentFarm, lineage is modeled as a **directed ancestry graph over agents** reconstructed from simulation snapshots:

- Snapshot records include parent references (`parent_ids`) and per-agent identity metadata (`agent_id`, generation, and chromosome values).
- The `farm.analysis.phylogenetics` module builds this ancestry into a lineage DAG (for example via `build_intrinsic_lineage_dag`).
- The library supports lineage-level analysis such as surviving lineage counts, depth over time, and lineage tree visualization (`plot_intrinsic_lineage_tree`).

In this library, lineage is an analysis structure used to track inheritance and diversification dynamics, not just a textual parent-child log.

## Phylogenetics

Phylogenetics is the study of evolutionary relationships among entities by reconstructing ancestry as branching trees or graphs.

In AgentFarm, phylogenetics is modeled as **lineage reconstruction and analysis over agent ancestry data**:

- The `farm.analysis.phylogenetics` module builds ancestry structures from snapshot records (for example with `build_intrinsic_lineage_dag`).
- Utilities support temporal lineage metrics such as surviving lineage counts and lineage depth over time.
- Visualization tools such as `plot_intrinsic_lineage_tree` render branching structure and can optionally color nodes by chromosome gene values.

In this library, phylogenetics is an analysis method for understanding inheritance structure and diversification patterns across simulated populations.

## Niche

A niche is the role or position a population occupies in its environment, including how it uses resources, responds to constraints, and interacts with competitors.

In AgentFarm, a niche is modeled as a **cluster-specific ecological profile** that can be measured from simulation telemetry:

- After clustering agents in chromosome space, the library can compute per-cluster environmental and behavioral summaries with `compute_niche_correlation`.
- These summaries include averages such as spatial location (`mean_x`, `mean_y`), energy (`mean_energy`), and reproduction cost (`mean_reproduction_cost`) for each detected cluster.
- This links genotype-like structure (chromosome clusters) to environment-embedded outcomes (where agents are, how costly reproduction is, and how they are performing).

In practice, a niche here means a recurring cluster pattern with distinguishable ecological context, not a fixed species label.

## Polymorphism

Polymorphism is the coexistence of two or more distinct inherited variants within a single population at the same time.

In AgentFarm, polymorphism is modeled as **durable variation across agent chromosomes during a single simulation run**:

- Agents in the same population can carry meaningfully different gene values (for example different `learning_rate` or `gamma`) that persist rather than collapsing to a single dominant form.
- Per-locus diversity metrics in `farm.analysis.genetics` (for example `compute_continuous_locus_diversity`, `compute_categorical_locus_diversity`, and allele-frequency trajectories) quantify how much variation is retained at each gene.
- Stable polymorphism shows up downstream as multiple chromosome clusters and an elevated, persistent `speciation_index` rather than a transient burst of variation that quickly converges.

In this library, polymorphism is the maintained gene-level variation that supports niche structure and speciation, not a fleeting variant in a converging population.

## Speciation

Speciation is the process by which a population splits into distinct sub-populations that remain meaningfully different over time (for example, because they adapt to different niches or selective pressures).

In AgentFarm, speciation is modeled as **cluster formation in hyperparameter chromosome space** during simulation runs:

- Agent chromosomes are represented as vectors of gene values (for example `learning_rate`, `gamma`, and `epsilon_decay`).
- The library detects sub-populations by clustering those vectors with either Gaussian Mixture Models (`detect_clusters_gmm`) or DBSCAN (`detect_clusters_dbscan`) in `farm.analysis.speciation`.
- It quantifies separation with a `speciation_index` in `[0, 1]` (computed from clustering structure; higher means clearer separation between groups).
- It tracks continuity of clusters across snapshot steps with centroid matching (`match_clusters_greedy`) and writes lineage records to `cluster_lineage.jsonl` when enabled through `GeneTrajectoryLogger`.

This definition is operational rather than biological: in this library, "species" means stable, separable chromosome niches within a single evolving simulation.