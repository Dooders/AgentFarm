

    @execute_query
    def execute(self, session) -> PopulationStatistics:
        #! this will basically be in population service
        """Calculate comprehensive population statistics for the entire simulation.

        Combines data from multiple analyses to create a complete statistical overview
        of the simulation's population dynamics, resource usage, and agent behavior.

        Returns
        -------
        PopulationStatistics
            Comprehensive statistics containing:
            - basic_stats : BasicPopulationStatistics
                Fundamental population metrics (avg, peak, steps, etc.)
            - resource_metrics : ResourceMetrics
                Resource consumption and utilization statistics
            - population_variance : PopulationVariance
                Statistical measures of population variation
            - agent_distribution : AgentDistribution
                Distribution of different agent types
            - survival_metrics : SurvivalMetrics
                Population survival rates and average lifespans

        Notes
        -----
        This method aggregates data from multiple queries and calculations to provide
        a complete statistical analysis of the simulation. If no data is available,
        it returns a PopulationStatistics object with zero values.
        """
        # Get base population data
        pop_data = self.population_data()

        # Get basic statistics
        basic_stats = self.basic_population_statistics(pop_data)
        if not basic_stats:
            return PopulationStatistics(
                population_metrics=PopulationMetrics(
                    total_agents=0,
                    system_agents=0,
                    independent_agents=0,
                    control_agents=0,
                ),
                population_variance=PopulationVariance(
                    variance=0.0, standard_deviation=0.0, coefficient_variation=0.0
                ),
            )

        # Calculate variance statistics
        variance = (basic_stats.sum_squared / basic_stats.step_count) - (
            basic_stats.avg_population**2
        )
        std_dev = variance**0.5
        cv = (
            std_dev / basic_stats.avg_population
            if basic_stats.avg_population > 0
            else 0
        )

        # Get agent type distribution
        type_stats = self.agent_type_distribution()

        # Create PopulationMetrics
        population_metrics = PopulationMetrics(
            total_agents=basic_stats.peak_population,
            system_agents=int(type_stats.system_agents),
            independent_agents=int(type_stats.independent_agents),
            control_agents=int(type_stats.control_agents),
        )

        # Create PopulationVariance
        population_variance = PopulationVariance(
            variance=variance, standard_deviation=std_dev, coefficient_variation=cv
        )

        # Return PopulationStatistics with the correct structure
        return PopulationStatistics(
            population_metrics=population_metrics,
            population_variance=population_variance,
        )
        
        
        
    @execute_query
    def basic_population_statistics(
        self, session, pop_data: Optional[List[Population]] = None
    ) -> BasicPopulationStatistics:
        """Calculate basic population statistics from step data.

        Processes raw population data to compute fundamental statistics about
        the population and resource usage.

        Parameters
        ----------
        pop_data : List[Population]
            List of Population objects containing step-wise simulation data

        Returns
        -------
        BasicPopulationStatistics
            Object containing:
            - avg_population : float
                Average population across all steps
            - death_step : int
                Final step number where agents existed
            - peak_population : int
                Maximum population reached
            - resources_consumed : float
                Total resources consumed across all steps
            - resources_available : float
                Total resources available across all steps
            - sum_squared : float
                Sum of squared population counts (for variance calculation)
            - step_count : int
                Total number of steps with active agents
        """
        if not pop_data:
            pop_data = self.population_data()

        # Calculate statistics directly from Population objects
        stats = {
            "avg_population": sum(p.total_agents for p in pop_data) / len(pop_data),
            "death_step": max(p.step_number for p in pop_data),
            "peak_population": max(p.total_agents for p in pop_data),
            "resources_consumed": sum(p.resources_consumed for p in pop_data),
            "resources_available": sum(p.total_resources for p in pop_data),
            "sum_squared": sum(p.total_agents * p.total_agents for p in pop_data),
            "step_count": len(pop_data),
        }

        return BasicPopulationStatistics(
            avg_population=float(stats["avg_population"] or 0),
            death_step=int(stats["death_step"] or 0),
            peak_population=int(stats["peak_population"] or 0),
            resources_consumed=float(stats["resources_consumed"] or 0),
            resources_available=float(stats["resources_available"] or 0),
            sum_squared=float(stats["sum_squared"] or 0),
            step_count=int(stats["step_count"] or 1),
        )
