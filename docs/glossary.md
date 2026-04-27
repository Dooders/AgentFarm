# Glossary

## Speciation

Speciation is the process by which a population splits into distinct sub-populations that remain meaningfully different over time (for example, because they adapt to different niches or selective pressures).

In AgentFarm, speciation is modeled as **cluster formation in hyperparameter chromosome space** during simulation runs:

- Agent chromosomes are represented as vectors of gene values (for example `learning_rate`, `gamma`, and `epsilon_decay`).
- The library detects sub-populations by clustering those vectors with either Gaussian Mixture Models (`detect_clusters_gmm`) or DBSCAN (`detect_clusters_dbscan`) in `farm.analysis.speciation`.
- It quantifies separation with a `speciation_index` in `[0, 1]` (computed from clustering structure; higher means clearer separation between groups).
- It tracks continuity of clusters across snapshot steps with centroid matching (`match_clusters_greedy`) and writes lineage records to `cluster_lineage.jsonl` when enabled through `GeneTrajectoryLogger`.

This definition is operational rather than biological: in this library, "species" means stable, separable chromosome niches within a single evolving simulation.

## Niche

A niche is the role or position a population occupies in its environment, including how it uses resources, responds to constraints, and interacts with competitors.

In AgentFarm, a niche is modeled as a **cluster-specific ecological profile** that can be measured from simulation telemetry:

- After clustering agents in chromosome space, the library can compute per-cluster environmental and behavioral summaries with `compute_niche_correlation`.
- These summaries include averages such as spatial location (`mean_x`, `mean_y`), energy (`mean_energy`), and reproduction cost (`mean_reproduction_cost`) for each detected cluster.
- This links genotype-like structure (chromosome clusters) to environment-embedded outcomes (where agents are, how costly reproduction is, and how they are performing).

In practice, a niche here means a recurring cluster pattern with distinguishable ecological context, not a fixed species label.

## Lineage

Lineage is the chain of descent connecting an individual or group to its ancestors over time.

In AgentFarm, lineage is modeled as a **directed ancestry graph over agents** reconstructed from simulation snapshots:

- Snapshot records include parent references (`parent_ids`) and per-agent identity metadata (`agent_id`, generation, and chromosome values).
- The `farm.analysis.phylogenetics` module builds this ancestry into a lineage DAG (for example via `build_intrinsic_lineage_dag`).
- The library supports lineage-level analysis such as surviving lineage counts, depth over time, and lineage tree visualization (`plot_intrinsic_lineage_tree`).

In this library, lineage is an analysis structure used to track inheritance and diversification dynamics, not just a textual parent-child log.

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

## Phylogenetics

Phylogenetics is the study of evolutionary relationships among entities by reconstructing ancestry as branching trees or graphs.

In AgentFarm, phylogenetics is modeled as **lineage reconstruction and analysis over agent ancestry data**:

- The `farm.analysis.phylogenetics` module builds ancestry structures from snapshot records (for example with `build_intrinsic_lineage_dag`).
- Utilities support temporal lineage metrics such as surviving lineage counts and lineage depth over time.
- Visualization tools such as `plot_intrinsic_lineage_tree` render branching structure and can optionally color nodes by chromosome gene values.

In this library, phylogenetics is an analysis method for understanding inheritance structure and diversification patterns across simulated populations.

## Genome

A genome is the complete set of inherited information that defines an individual.

In AgentFarm, a genome is modeled as a **serializable agent-level configuration and ancestry representation**:

- The `farm.core.genome` module captures and reconstructs agent state/configuration through a genome dictionary interface.
- `genome_id` fields are used across storage and analysis layers to encode ancestry relationships and lineage grouping.
- Genome-level operations (such as mutation and crossover in the genome utilities) support evolutionary experimentation on inherited agent traits.

In this library, genome refers to computationally encoded inherited agent information, not a biochemical DNA sequence.

## Inheritance

Inheritance is the transfer of information or traits from parent(s) to offspring.

In AgentFarm, inheritance is modeled as **offspring initialization from parent-derived genome/chromosome information**:

- Reproduction paths pass parent-derived values into child configuration and learning parameters.
- Parent relationships are tracked through lineage metadata (for example `parent_ids` and `genome_id` conventions).
- Inheritance can be direct (copy-like) or combined with crossover and mutation, depending on the active reproduction and evolution settings.

In this library, inheritance is the mechanism that preserves continuity across generations in simulated populations.

## Mutation

Mutation is a random or stochastic change to inherited information that introduces variation.

In AgentFarm, mutation is modeled as **parameter perturbation during genetic operations**:

- Hyperparameter chromosomes support mutation operators and modes (for example Gaussian and multiplicative behavior in the chromosome utilities).
- Genome utilities also support mutation of inherited configuration structures.
- Mutation rates and scales control exploration versus stability during evolutionary runs.

In practice, mutation is the primary source of novel variation in the library's inheritance pipeline.

## Crossover

Crossover is the recombination of inherited information from multiple parents to form offspring.

In AgentFarm, crossover is modeled as **operator-driven recombination of parent chromosome/genome data**:

- Hyperparameter chromosome utilities provide multiple crossover modes (for example single-point, uniform, blend, and multi-point).
- Crossover can be enabled or disabled by policy/config and is applied during reproduction when eligible co-parents are available.
- The resulting child representation is then optionally followed by mutation, forming a full recombination-plus-variation step.

In this library, crossover increases diversity by mixing existing inherited structures rather than creating variation from scratch.

## Reproduction

Reproduction is the process by which new individuals are created from existing parent individuals.

In AgentFarm, reproduction is modeled as **offspring creation with inherited and optionally transformed parameters**:

- Reproduction uses parent information to initialize child state and inherited representations (for example chromosome/genome-linked fields).
- Depending on configuration and eligibility, offspring generation can include direct inheritance, crossover with a co-parent, and mutation.
- Reproduction outcomes feed lineage tracking through ancestry metadata and can change population structure over time.

In this library, reproduction is the generational transition mechanism that connects survival dynamics to evolutionary change.

## Energy

Energy is a quantitative resource reserve that supports activity, maintenance, and survival.

In AgentFarm, energy is modeled as an **agent state variable that mediates behavior and lifecycle outcomes**:

- Agents spend energy while acting and can gain energy through successful resource interactions.
- Energy level influences survival pressure and can constrain whether agents persist long enough to reproduce.
- Energy-related telemetry (for example `mean_energy`) is used in niche and population analyses to connect behavioral/ecological context to evolutionary outcomes.

In this library, energy is both a local control signal for individual agents and a global driver of population-level dynamics.

## Agent

An agent is an autonomous entity that perceives its environment, makes decisions, and takes actions toward survival or other objectives.

In AgentFarm, an agent is modeled as a **stateful simulation actor with configurable behavior, resources, and lifecycle dynamics**:

- Agents hold internal state (for example energy, generation, and ancestry metadata) and interact with the environment through action systems.
- Agent behavior can be driven by configured modules and learning-related parameters, which may be inherited and transformed across generations.
- Agent interactions with resources, space, and other agents generate the population-level patterns analyzed throughout the library.

In this library, agents are the core units of simulation from which lineage, ecological dynamics, and emergent behavior are derived.

