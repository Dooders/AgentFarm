# Data & Visualization

## Overview

AgentFarm provides a comprehensive data collection, analysis, and visualization framework that transforms raw simulation data into meaningful insights. The system automatically tracks an extensive array of metrics throughout simulation execution while providing flexible, powerful tools for visual exploration and automated report generation. This integrated approach to data handling ensures that no interesting behavior goes unnoticed and that researchers can effectively communicate their findings through compelling visualizations and comprehensive reports.

The data and visualization infrastructure in AgentFarm recognizes that understanding complex simulation dynamics requires multiple perspectives and levels of analysis. Simple summary statistics might hide important details visible only through time series plots. Aggregate population metrics might obscure interesting individual variation visible in scatter plots or network visualizations. Spatial patterns might not be apparent without heatmaps or animated visualizations showing agent movement. The platform provides all these views and more, enabling researchers to build a multi-faceted understanding of their simulations.

## Comprehensive Automated Data Collection

At the foundation of AgentFarm's data capabilities is a robust automatic data collection system that runs continuously during simulations, capturing detailed information about every aspect of system state and dynamics. This automatic collection ensures that data is never lost due to oversight and that all simulations produce comparable datasets suitable for systematic analysis. The collection system is designed to be both comprehensive and efficient, capturing rich detail without significantly impacting simulation performance.

Agent-level data collection tracks the state of every agent at each timestep, recording health levels, resource inventories, spatial positions and movement vectors, current actions and action histories, social relationships and interaction counts, learning progress and accumulated experience, and any custom attributes specific to your agent types. This granular agent-level data enables detailed analysis of individual trajectories, identification of exceptional individuals, and understanding of within-population variation.

Population-level metrics aggregate agent data to characterize the population as a whole. The system tracks total population size and birth/death rates, demographic distributions across age, generation, and type, summary statistics of agent attributes like mean, variance, and extremes, diversity measures that quantify population heterogeneity, and spatial distributions that describe how populations spread across the environment. These population-level views are essential for understanding macro-level dynamics and emergent phenomena.

Environmental data collection captures the state of the environment itself, including resource levels across different locations, environmental quality metrics, spatial gradients and patterns, and histories of environmental change. This environmental tracking enables analysis of agent-environment feedback loops, resource depletion dynamics, and how environmental heterogeneity influences population structure and behavior.

Interaction data represents a rich source of information about the social structure and dynamics of agent populations. Every interaction between agents is recorded with detailed metadata including the agents involved and their states, the type of interaction that occurred, the context and location of the interaction, and the outcomes for each participant. This comprehensive interaction logging enables network analysis, identification of keystone individuals, and understanding of how social structure emerges and evolves.

Time-series data collection maintains longitudinal records of all key metrics, creating datasets suitable for temporal analysis. These time series reveal trends, cycles, and transitions that characterize simulation dynamics. The temporal resolution of data collection can be configured to balance detail against storage requirements, with options to collect data every timestep, at regular intervals, or at specific events of interest.

## Real-Time Visualization Capabilities

For interactive exploration and monitoring of ongoing simulations, AgentFarm provides real-time visualization capabilities that update continuously as simulations run. Real-time visualization serves multiple purposes including debugging and validation where you can quickly spot problems, hypothesis generation as patterns suggest new questions, and presentation where live simulations create engaging demonstrations of complex system dynamics.

The real-time spatial visualization displays agent positions and movements in an animated view of the simulation world. Agents can be color-coded by attributes like health, resources, or type, allowing you to see at a glance how these properties distribute across space. Movement trails can visualize recent trajectories, revealing spatial strategies like territoriality or migration. Resource distributions can be overlaid as heatmaps, showing how agents respond to environmental heterogeneity. This spatial view provides immediate intuition about system dynamics that would be difficult to extract from numerical data alone.

Real-time metric plots display time series of key indicators updating as the simulation progresses. You can monitor population size, average fitness, resource depletion, diversity indices, and any custom metrics you've defined. Multiple metrics can be plotted simultaneously to reveal relationships and tradeoffs. Threshold lines and reference values help assess whether simulations are behaving as expected or revealing surprising dynamics worth investigating further.

Interactive controls allow you to adjust visualization parameters on the fly, zoom in on specific regions or time periods, pause and resume simulation, and even modify parameters while simulations run for exploratory "what-if" experiments. This interactivity makes real-time visualization a powerful tool for developing intuition about model behavior and for calibrating models to produce realistic dynamics.

## Static Visualization and Analysis

After simulations complete, comprehensive static visualization tools enable detailed analysis of simulation outcomes and the creation of publication-quality figures. Static visualizations can be more sophisticated than real-time displays because computation time is less constrained, allowing for complex layouts, statistical annotations, and careful aesthetic refinement.

Spatial visualization tools create detailed maps of agent distributions, spatial patterns, and environmental features at specific timepoints or averaged over periods. Agents can be represented as points, with size and color encoding multiple attributes simultaneously. Density heatmaps reveal clustering and avoidance patterns. Spatial statistics can be overlaid to test for significant deviations from random distributions. Multiple spatial snapshots can be arranged into figure panels showing how patterns evolve over time.

Population timeline visualizations chart how population properties change over the course of simulations. Simple line plots track single metrics, while stacked area charts show how different components contribute to totals. Confidence bands can represent variation across replications. Event markers highlight significant occurrences like population crashes, invasions, or transitions. These temporal visualizations are essential for understanding simulation dynamics and identifying phases of simulation behavior.

Resource distribution plots show how resources spread across space and how their distribution changes over time. Heatmaps visualize spatial patterns, while time series show total availability and depletion rates. Resource-agent overlay plots reveal how agent distributions respond to resource availability, showing whether agents track resources as expected or exhibit more complex spatial strategies.

## Network Visualization and Analysis

Given the importance of interactions in agent-based models, AgentFarm provides specialized tools for constructing and visualizing interaction networks. Network visualization transforms interaction data into graph representations where nodes represent agents and edges represent interactions, revealing social structure and patterns of influence that might not be apparent from individual interaction records.

Interaction networks can be constructed using various criteria including all interactions between agents, only strong or frequent interactions, or interactions of specific types like cooperation or competition. Edge weights can represent interaction frequency, strength, or outcomes. Node attributes like size and color can encode agent properties, creating rich network visualizations that integrate social structure with agent characteristics.

Social network analysis metrics are computed automatically including degree centrality measuring how connected each agent is, betweenness centrality identifying agents that bridge different groups, clustering coefficients quantifying local neighborhood connectivity, and community detection revealing subgroup structure. These metrics provide quantitative characterizations of network structure that complement visual inspection.

Network layout algorithms position nodes to reveal structure, using force-directed layouts for general networks, hierarchical layouts for networks with clear stratification, circular layouts for highlighting cycles and symmetry, and spatial layouts that preserve geographic positions when relevant. The choice of layout algorithm can dramatically affect what patterns are visible, and AgentFarm provides multiple options to ensure you can find views that best reveal your data's structure.

Temporal network visualization shows how social structure evolves over time. Animated network displays show edges appearing and disappearing as relationships form and dissolve. Static panel figures capture network structure at key timepoints. Stability analysis measures how much network structure changes between timesteps. These temporal network analyses reveal whether social structure is static, slowly evolving, or highly dynamic.

## Comprehensive Charting and Plotting

Beyond specialized visualizations, AgentFarm provides general-purpose charting tools for creating standard statistical plots and exploratory data analysis visualizations. These tools integrate seamlessly with the simulation data infrastructure, making it easy to generate a wide variety of plots without extensive data manipulation.

Time-series charts are perhaps the most fundamental visualization for simulation data. Line charts track how metrics evolve over time, revealing trends, cycles, and transitions. Multi-line charts compare multiple metrics or different experimental conditions on the same axes. Area charts emphasize magnitudes and can be stacked to show component contributions. Confidence bands and shading indicate uncertainty across replications. Time-series decomposition separates trend, seasonal, and residual components when appropriate.

Distribution plots characterize the spread and shape of data distributions at specific timepoints. Histograms bin continuous data to show distribution shape. Kernel density estimates provide smoothed distribution representations. Box plots compactly summarize distributions with quartiles and outliers. Violin plots combine box plots with density estimates for richer summaries. These distribution plots are essential for understanding population heterogeneity and identifying outliers.

Scatter plots explore relationships between pairs of variables, revealing correlations, clusters, and nonlinear dependencies. Points can be color-coded by third variables, creating trivariate visualizations. Trend lines and confidence regions summarize relationships statistically. Interactive scatter plots allow brushing and linking across multiple views, enabling exploratory multivariate analysis.

Correlation matrices and heatmaps visualize relationships across many variables simultaneously. Correlation heatmaps use color to encode correlation strength, making patterns of covariation immediately apparent. Hierarchical clustering can reorder variables to group related measures together. These high-level visualizations help identify key relationships and guide more detailed analyses.

Comparison plots facilitate evaluation of experimental treatments or parameter variations. Bar charts compare outcomes across conditions, with error bars indicating uncertainty. Grouped or stacked bars compare multiple metrics simultaneously. Difference plots and effect size displays emphasize the magnitude of differences rather than raw values. Statistical annotations can indicate significance of differences automatically.

## Automated Report Generation

To streamline the process of documenting and communicating simulation results, AgentFarm includes automated report generation capabilities that combine visualizations, statistical summaries, and narrative text into comprehensive documents. Report generation serves both personal documentation needs and preparation of materials for publication or presentation.

Summary reports provide high-level overviews of simulation outcomes suitable for quick assessment or executive summaries. These reports include key metrics and their values, population dynamics visualizations, significant events and transitions, comparisons to baselines or previous results, and methodological notes documenting configuration settings. Summary reports can be generated automatically for every simulation, creating a searchable archive of experimental results.

Analysis reports dive deeper into simulation dynamics with detailed statistical analysis, extensive visualization suites, tests of specific hypotheses, and discussion of findings. These reports are customizable through templates that specify which analyses to perform, how to visualize results, and what statistical tests to apply. Report templates promote consistency across analyses and encode best practices for analyzing specific model types.

Comparison reports facilitate systematic evaluation of parameter effects or experimental treatments. These reports present results from multiple simulation runs in parallel, with side-by-side visualizations highlighting differences, statistical tests assessing significance of effects, and summary tables quantifying effect sizes. The comparison report format makes it easy to assess which parameters matter and how they influence outcomes.

Custom reports can be constructed programmatically for specialized needs, combining automated elements with hand-crafted analysis and interpretation. The report generation system exposes a flexible API that allows you to mix standard report components with custom analysis code, insert custom visualizations, include arbitrary text and formatting, and export to various formats including HTML, PDF, and Markdown.

## Multiple Export Formats

AgentFarm supports exporting simulation data in various formats to facilitate analysis with external tools and sharing with collaborators. Format selection depends on data volume, intended use, and tool compatibility. The platform handles format conversion automatically, shielding users from the complexity of different file formats.

CSV export provides a simple, universal format for tabular data that is readable by virtually every analysis tool. The CSV exporter can flatten complex hierarchical data into tabular form, handle missing values appropriately, and include metadata headers. While CSV is simple and universal, it can be inefficient for very large datasets and doesn't preserve data types perfectly.

JSON export creates hierarchical, self-describing data files that preserve structure and data types. JSON is ideal for complex nested data, configuration files, and interchange with web-based tools. The format is human-readable for inspection and debugging while still being efficiently machine-parseable. JSON export includes full metadata about simulation configuration and data provenance.

HDF5 export provides efficient storage for very large simulation datasets. HDF5 is a binary format optimized for scientific computing, supporting compression, chunking for efficient partial access, and datasets larger than available memory. The format is particularly valuable for large-scale experiments where storage efficiency and access performance matter. AgentFarm's HDF5 export includes appropriate chunking and compression by default while allowing customization for advanced users.

Database export writes simulation data to SQLite databases, enabling efficient querying and analysis using SQL. Database export is valuable for large datasets where you want to perform complex queries, aggregate analyses, and filtering without loading all data into memory. The database format also facilitates integration with database-backed analysis tools and reproducible analysis pipelines.

## Interactive Dashboards and Exploration

For deep exploratory analysis, AgentFarm provides interactive dashboard capabilities that combine multiple coordinated visualizations with interactive controls. Dashboards enable researchers to explore simulation data from multiple angles simultaneously, with interactions in one view updating others to maintain consistency.

Jupyter notebook integration brings AgentFarm's visualization capabilities into the rich ecosystem of interactive Python notebooks. Within notebooks, you can create interactive widgets that control visualization parameters, combine code, visualizations, and narrative text in a single document, and export notebooks as reproducible analysis documents. This integration makes Jupyter notebooks an ideal environment for exploratory analysis and for creating analysis tutorials or teaching materials.

Web-based dashboards provide rich interactive experiences accessible through web browsers without requiring local software installation. These dashboards can include coordinated multiple views where selections propagate across visualizations, interactive filtering and querying of data, animated playback of simulation dynamics, and real-time updates when analyzing ongoing simulations. Web dashboards are valuable for collaboration and for creating shareable analysis applications.

## Performance Optimization for Visualization

Visualizing large simulation datasets presents performance challenges that AgentFarm addresses through several optimization strategies. These optimizations ensure that visualization remains responsive even when dealing with millions of datapoints or long simulation runs.

Data downsampling reduces the number of points plotted when full resolution exceeds what can be meaningfully displayed or would slow rendering unacceptably. Intelligent downsampling preserves important features like peaks, outliers, and trends while reducing data volume. The platform can automatically determine appropriate downsampling levels based on display resolution and data characteristics.

Level-of-detail rendering adjusts visualization detail based on zoom level and viewing context. When viewing an entire simulation at once, simplified representations suffice. When zooming in on specific regions or timepoints, full detail becomes visible. This adaptive approach keeps visualization responsive across scales.

Caching of rendered visualizations avoids redundant computation when repeatedly viewing the same data. Computed plots are cached and reused when visualization parameters haven't changed. This caching is particularly valuable for expensive computations like network layouts or complex statistical graphics.

Progressive rendering displays incremental results while completing expensive visualizations, providing immediate feedback even for slow operations. Initial approximations appear quickly, then refine as computation proceeds. This approach keeps interfaces responsive and allows users to interrupt long-running operations if needed.

## Best Practices

Effective visualization requires thoughtful choices about what to display and how to display it. Follow visualization best practices including choosing appropriate plot types for your data types, using color effectively to highlight important distinctions without overwhelming, including proper axis labels, legends, and titles, providing uncertainty estimates when appropriate, and ensuring that visualizations are accessible to audiences with varying needs.

Data-ink ratio optimization, a principle from information design, suggests maximizing the proportion of visualization devoted to representing data rather than decoration. Remove unnecessary gridlines, borders, and embellishments. Use simple, clean designs that direct attention to the data itself. This minimalist approach produces clearer, more effective visualizations.

Reproducibility in visualization is important for scientific work. Save visualization code along with data so that figures can be regenerated if needed. Document any manual adjustments or customizations applied. Use deterministic color schemes and layouts rather than random selections. These practices ensure that visualizations can be recreated and verified by others.

## Related Documentation

For more details on specific aspects of data and visualization, consult the Data System documentation which describes the underlying data architecture and storage. The Data Services documentation explains programmatic access to simulation data. The Data Retrieval guide covers querying and filtering operations. The Metrics documentation describes built-in and custom metric definitions. The Analysis documentation details statistical analysis capabilities that complement visualization.

## Examples and Case Studies

Practical examples of data analysis and visualization can be found throughout the documentation. Usage Examples demonstrate common visualization patterns. Service Usage Examples show how to access and prepare data for visualization. Experiment Analysis documentation illustrates complete analytical workflows from data extraction through visualization to interpretation. The One of a Kind Experiments provide rich examples of visualization in the context of specific research questions, showing how visualization choices support scientific inference and communication.
