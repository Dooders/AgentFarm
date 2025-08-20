#!/usr/bin/env python3
"""
Alien Invasion Combat Simulation Runner

This script runs an alien invasion scenario where aliens spawn around the edges 
and humans defend from the center. The simulation features:

- Strategic spawn positioning
- Faction-based combat bonuses
- Territory control mechanics
- Enhanced combat analytics
"""

import logging
import sys
import os
from pathlib import Path

# Add farm module to path
sys.path.insert(0, str(Path(__file__).parent))

from farm.core.config import SimulationConfig
from farm.environments.alien_invasion_environment import create_alien_invasion_simulation
from farm.actions.invasion_attack import get_invasion_attack_stats
from farm.database.database import SimulationDatabase


def setup_logging():
    """Setup logging for the alien invasion simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('alien_invasion.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_alien_invasion_simulation(
    config_path: str = "config_alien_invasion.yaml",
    num_steps: int = 2000,
    output_dir: str = "results/alien_invasion"
) -> dict:
    """Run the alien invasion simulation with comprehensive tracking.
    
    Parameters
    ----------
    config_path : str
        Path to the alien invasion configuration file
    num_steps : int
        Number of simulation steps to run
    output_dir : str
        Directory to save simulation results
        
    Returns
    -------
    dict
        Comprehensive simulation results and statistics
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Alien Invasion Simulation")
    logger.info(f"Config: {config_path}")
    logger.info(f"Steps: {num_steps}")
    
    try:
        # Load configuration
        config = SimulationConfig.from_yaml(config_path)
        config.simulation_steps = num_steps
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create alien invasion environment
        environment = create_alien_invasion_simulation(config)
        
        logger.info(f"Environment created with {len(environment.agents)} agents")
        logger.info(f"Humans: {config.human_agents}, Aliens: {config.alien_agents}")
        
        # Track simulation metrics
        step_data = []
        combat_history = []
        
        # Run simulation
        for step in range(num_steps):
            environment.time = step
            
            # Log periodic status
            if step % 100 == 0:
                status = environment.get_invasion_status()
                logger.info(f"Step {step}: Humans={status['humans_alive']}, "
                           f"Aliens={status['aliens_alive']}, "
                           f"Territory={status['territorial_control']['human_ratio']:.2f}")
                
                # Check for early victory
                victory, message = environment.check_victory_conditions()
                if victory:
                    logger.info(f"Victory achieved at step {step}: {message}")
                    break
            
            # Save step data every 50 steps
            if step % 50 == 0:
                step_metrics = environment._calculate_metrics()
                step_data.append({
                    'step': step,
                    'metrics': step_metrics,
                    'invasion_status': environment.get_invasion_status()
                })
            
            # Save combat statistics
            if step % 20 == 0:
                combat_stats = get_invasion_attack_stats(environment)
                combat_history.append({
                    'step': step,
                    'stats': combat_stats
                })
            
            # Execute agent actions
            for agent in environment.agents[:]:  # Use slice to avoid modification during iteration
                if agent.alive:
                    agent.act()
            
            # Update environment
            environment.update()
            
            # Check for population extinction
            living_agents = [a for a in environment.agents if a.alive]
            if len(living_agents) == 0:
                logger.warning(f"All agents died at step {step}")
                break
        
        # Collect final results
        final_status = environment.get_invasion_status()
        final_combat_stats = get_invasion_attack_stats(environment)
        victory, victory_message = environment.check_victory_conditions()
        
        # Compile comprehensive results
        results = {
            'simulation_completed': True,
            'total_steps': step + 1,
            'victory_achieved': victory,
            'victory_message': victory_message,
            'final_status': final_status,
            'final_combat_stats': final_combat_stats,
            'step_data': step_data,
            'combat_history': combat_history,
            'config': config.to_dict(),
            'database_path': environment.db.db_path if environment.db else None,
        }
        
        # Save results summary
        import json
        results_file = Path(output_dir) / "invasion_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Simulation completed successfully")
        logger.info(f"Results saved to: {results_file}")
        
        # Cleanup
        if environment.db:
            environment.db.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise


def analyze_invasion_results(results: dict) -> dict:
    """Analyze the simulation results and provide insights.
    
    Parameters
    ----------
    results : dict
        Results from run_alien_invasion_simulation
        
    Returns
    -------
    dict
        Analysis and insights from the simulation
    """
    logger = logging.getLogger(__name__)
    
    final_status = results['final_status']
    combat_stats = results['final_combat_stats']
    
    # Basic analysis
    analysis = {
        'survival_analysis': {
            'human_survival_rate': final_status['humans_alive'] / results['config']['human_agents'],
            'alien_survival_rate': final_status['aliens_alive'] / results['config']['alien_agents'],
            'total_eliminations': final_status['humans_eliminated'] + final_status['aliens_eliminated'],
        },
        'combat_analysis': {
            'combat_intensity': combat_stats['combat_intensity'],
            'attack_success_rate': combat_stats['successful_attacks'] / max(1, combat_stats['combat_encounters']),
            'average_combat_per_step': combat_stats['combat_encounters'] / results['total_steps'],
        },
        'territorial_analysis': {
            'final_control_ratio': final_status['territorial_control']['human_ratio'],
            'territory_shift': final_status['territorial_control']['human_ratio'] - 0.5,  # From neutral
        },
        'victory_analysis': {
            'victory_achieved': results['victory_achieved'],
            'victory_type': results['victory_message'],
            'simulation_duration': results['total_steps'],
        }
    }
    
    # Calculate trends from step data
    if results['step_data']:
        # Territory control trend
        territory_data = [step['invasion_status']['territorial_control']['human_ratio'] 
                         for step in results['step_data']]
        analysis['trends'] = {
            'territory_trend': 'increasing' if territory_data[-1] > territory_data[0] else 'decreasing',
            'territory_volatility': _calculate_volatility(territory_data),
        }
    
    logger.info("Analysis completed")
    for category, metrics in analysis.items():
        logger.info(f"{category}: {metrics}")
    
    return analysis


def _calculate_volatility(data: list) -> float:
    """Calculate volatility (standard deviation) of a data series."""
    if len(data) < 2:
        return 0.0
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Alien Invasion Combat Simulation")
    parser.add_argument("--config", default="config_alien_invasion.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--steps", type=int, default=2000,
                       help="Number of simulation steps")
    parser.add_argument("--output", default="results/alien_invasion",
                       help="Output directory for results")
    parser.add_argument("--analyze", action="store_true",
                       help="Run analysis after simulation")
    
    args = parser.parse_args()
    
    try:
        # Run simulation
        results = run_alien_invasion_simulation(
            config_path=args.config,
            num_steps=args.steps,
            output_dir=args.output
        )
        
        print(f"\n{'='*60}")
        print("ALIEN INVASION SIMULATION COMPLETED")
        print(f"{'='*60}")
        print(f"Victory: {results['victory_message']}")
        print(f"Duration: {results['total_steps']} steps")
        print(f"Final Status: {results['final_status']}")
        
        # Run analysis if requested
        if args.analyze:
            print(f"\n{'='*60}")
            print("RUNNING ANALYSIS")
            print(f"{'='*60}")
            analysis = analyze_invasion_results(results)
            
            # Save analysis
            import json
            analysis_file = Path(args.output) / "invasion_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"Analysis saved to: {analysis_file}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)