def run_simulation(
    num_steps: int,
    config: SimulationConfig,
    experiment_id: Optional[str] = None,
    path: Optional[str] = None,
) -> Environment:
    """Run a single simulation.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps to run
    config : SimulationConfig
        Configuration for the simulation
    experiment_id : Optional[str]
        ID of the experiment this simulation belongs to
    path : Optional[str]
        Path to store simulation results

    Returns
    -------
    Environment
        The simulation environment after running
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing simulation")

    # Setup database
    db = SimulationDatabase(path) if path else None

    try:
        # Initialize environment
        environment = Environment(
            config=config,
            database=db,
            experiment_id=experiment_id
        )

        # Run simulation steps
        for step in range(num_steps):
            environment.step()

            # Log progress periodically
            if step > 0 and step % 1000 == 0:
                logger.info(f"Completed {step} steps")

        logger.info("Simulation completed successfully")
        return environment

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise 