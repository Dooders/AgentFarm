import os
import pandas as pd
import logging
from sqlalchemy.orm import Session

from sqlalchemy_models import (
    init_db, get_session, Simulation, DominanceMetrics, AgentPopulation,
    ReproductionStats, DominanceSwitching, ResourceDistribution,
    HighLowSwitchingComparison, CorrelationAnalysis
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def import_csv_to_db(csv_path, db_path='sqlite:///dominance.db'):
    """
    Import data from the dominance_analysis.csv file into the SQLAlchemy database.
    
    Args:
        csv_path: Path to the dominance_analysis.csv file
        db_path: SQLAlchemy database URL
    """
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return False
    
    # Read CSV file
    logging.info(f"Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return False
    
    # Initialize database
    logging.info(f"Initializing database: {db_path}")
    engine = init_db(db_path)
    session = get_session(engine)
    
    # Import data row by row
    logging.info("Importing data into database...")
    try:
        for _, row in df.iterrows():
            # Create Simulation record
            sim = Simulation(iteration=row['iteration'])
            session.add(sim)
            session.flush()  # Flush to get the ID
            
            # Create DominanceMetrics record
            dominance_metrics = DominanceMetrics(
                simulation_id=sim.id,
                population_dominance=row['population_dominance'],
                survival_dominance=row['survival_dominance'],
                comprehensive_dominance=row['comprehensive_dominance'],
                system_dominance_score=row['system_dominance_score'],
                independent_dominance_score=row['independent_dominance_score'],
                control_dominance_score=row['control_dominance_score'],
                system_auc=row['system_auc'],
                independent_auc=row['independent_auc'],
                control_auc=row['control_auc'],
                system_recency_weighted_auc=row['system_recency_weighted_auc'],
                independent_recency_weighted_auc=row['independent_recency_weighted_auc'],
                control_recency_weighted_auc=row['control_recency_weighted_auc'],
                system_dominance_duration=row['system_dominance_duration'],
                independent_dominance_duration=row['independent_dominance_duration'],
                control_dominance_duration=row['control_dominance_duration'],
                system_growth_trend=row['system_growth_trend'],
                independent_growth_trend=row['independent_growth_trend'],
                control_growth_trend=row['control_growth_trend'],
                system_final_ratio=row['system_final_ratio'],
                independent_final_ratio=row['independent_final_ratio'],
                control_final_ratio=row['control_final_ratio']
            )
            session.add(dominance_metrics)
            
            # Create AgentPopulation record
            agent_population = AgentPopulation(
                simulation_id=sim.id,
                system_agents=row.get('system_agents'),
                independent_agents=row.get('independent_agents'),
                control_agents=row.get('control_agents'),
                total_agents=row.get('total_agents'),
                final_step=row.get('final_step'),
                system_count=row.get('system_count'),
                system_alive=row.get('system_alive'),
                system_dead=row.get('system_dead'),
                system_avg_survival=row.get('system_avg_survival'),
                system_dead_ratio=row.get('system_dead_ratio'),
                independent_count=row.get('independent_count'),
                independent_alive=row.get('independent_alive'),
                independent_dead=row.get('independent_dead'),
                independent_avg_survival=row.get('independent_avg_survival'),
                independent_dead_ratio=row.get('independent_dead_ratio'),
                control_count=row.get('control_count'),
                control_alive=row.get('control_alive'),
                control_dead=row.get('control_dead'),
                control_avg_survival=row.get('control_avg_survival'),
                control_dead_ratio=row.get('control_dead_ratio'),
                initial_system_count=row.get('initial_system_count'),
                initial_independent_count=row.get('initial_independent_count'),
                initial_control_count=row.get('initial_control_count'),
                initial_resource_count=row.get('initial_resource_count'),
                initial_resource_amount=row.get('initial_resource_amount')
            )
            session.add(agent_population)
            
            # Create ReproductionStats record
            reproduction_stats = ReproductionStats(
                simulation_id=sim.id,
                system_reproduction_attempts=row.get('system_reproduction_attempts'),
                system_reproduction_successes=row.get('system_reproduction_successes'),
                system_reproduction_failures=row.get('system_reproduction_failures'),
                system_reproduction_success_rate=row.get('system_reproduction_success_rate'),
                system_first_reproduction_time=row.get('system_first_reproduction_time'),
                system_reproduction_efficiency=row.get('system_reproduction_efficiency'),
                system_avg_resources_per_reproduction=row.get('system_avg_resources_per_reproduction'),
                system_avg_offspring_resources=row.get('system_avg_offspring_resources'),
                independent_reproduction_attempts=row.get('independent_reproduction_attempts'),
                independent_reproduction_successes=row.get('independent_reproduction_successes'),
                independent_reproduction_failures=row.get('independent_reproduction_failures'),
                independent_reproduction_success_rate=row.get('independent_reproduction_success_rate'),
                independent_first_reproduction_time=row.get('independent_first_reproduction_time'),
                independent_reproduction_efficiency=row.get('independent_reproduction_efficiency'),
                independent_avg_resources_per_reproduction=row.get('independent_avg_resources_per_reproduction'),
                independent_avg_offspring_resources=row.get('independent_avg_offspring_resources'),
                control_reproduction_attempts=row.get('control_reproduction_attempts'),
                control_reproduction_successes=row.get('control_reproduction_successes'),
                control_reproduction_failures=row.get('control_reproduction_failures'),
                control_reproduction_success_rate=row.get('control_reproduction_success_rate'),
                control_first_reproduction_time=row.get('control_first_reproduction_time'),
                control_reproduction_efficiency=row.get('control_reproduction_efficiency'),
                control_avg_resources_per_reproduction=row.get('control_avg_resources_per_reproduction'),
                control_avg_offspring_resources=row.get('control_avg_offspring_resources'),
                independent_vs_control_first_reproduction_advantage=row.get('independent_vs_control_first_reproduction_advantage'),
                independent_vs_control_reproduction_efficiency_advantage=row.get('independent_vs_control_reproduction_efficiency_advantage'),
                independent_vs_control_reproduction_rate_advantage=row.get('independent_vs_control_reproduction_rate_advantage'),
                system_vs_independent_reproduction_rate_advantage=row.get('system_vs_independent_reproduction_rate_advantage'),
                system_vs_control_reproduction_rate_advantage=row.get('system_vs_control_reproduction_rate_advantage'),
                system_vs_independent_reproduction_efficiency_advantage=row.get('system_vs_independent_reproduction_efficiency_advantage'),
                system_vs_control_first_reproduction_advantage=row.get('system_vs_control_first_reproduction_advantage'),
                system_vs_independent_first_reproduction_advantage=row.get('system_vs_independent_first_reproduction_advantage'),
                system_vs_control_reproduction_efficiency_advantage=row.get('system_vs_control_reproduction_efficiency_advantage')
            )
            session.add(reproduction_stats)
            
            # Create DominanceSwitching record
            dominance_switching = DominanceSwitching(
                simulation_id=sim.id,
                total_switches=row.get('total_switches'),
                switches_per_step=row.get('switches_per_step'),
                dominance_stability=row.get('dominance_stability'),
                system_avg_dominance_period=row.get('system_avg_dominance_period'),
                independent_avg_dominance_period=row.get('independent_avg_dominance_period'),
                control_avg_dominance_period=row.get('control_avg_dominance_period'),
                early_phase_switches=row.get('early_phase_switches'),
                middle_phase_switches=row.get('middle_phase_switches'),
                late_phase_switches=row.get('late_phase_switches'),
                control_avg_switches=row.get('control_avg_switches'),
                independent_avg_switches=row.get('independent_avg_switches'),
                system_avg_switches=row.get('system_avg_switches'),
                system_to_system=row.get('system_to_system'),
                system_to_independent=row.get('system_to_independent'),
                system_to_control=row.get('system_to_control'),
                independent_to_system=row.get('independent_to_system'),
                independent_to_independent=row.get('independent_to_independent'),
                independent_to_control=row.get('independent_to_control'),
                control_to_system=row.get('control_to_system'),
                control_to_independent=row.get('control_to_independent'),
                control_to_control=row.get('control_to_control')
            )
            session.add(dominance_switching)
            
            # Create ResourceDistribution record
            resource_distribution = ResourceDistribution(
                simulation_id=sim.id,
                systemagent_avg_resource_dist=row.get('systemagent_avg_resource_dist'),
                systemagent_weighted_resource_dist=row.get('systemagent_weighted_resource_dist'),
                systemagent_nearest_resource_dist=row.get('systemagent_nearest_resource_dist'),
                systemagent_resources_in_range=row.get('systemagent_resources_in_range'),
                systemagent_resource_amount_in_range=row.get('systemagent_resource_amount_in_range'),
                independentagent_avg_resource_dist=row.get('independentagent_avg_resource_dist'),
                independentagent_weighted_resource_dist=row.get('independentagent_weighted_resource_dist'),
                independentagent_nearest_resource_dist=row.get('independentagent_nearest_resource_dist'),
                independentagent_resources_in_range=row.get('independentagent_resources_in_range'),
                independentagent_resource_amount_in_range=row.get('independentagent_resource_amount_in_range'),
                controlagent_avg_resource_dist=row.get('controlagent_avg_resource_dist'),
                controlagent_weighted_resource_dist=row.get('controlagent_weighted_resource_dist'),
                controlagent_nearest_resource_dist=row.get('controlagent_nearest_resource_dist'),
                controlagent_resources_in_range=row.get('controlagent_resources_in_range'),
                controlagent_resource_amount_in_range=row.get('controlagent_resource_amount_in_range'),
                positive_corr_controlagent_resource_amount_in_range=row.get('positive_corr_controlagent_resource_amount_in_range'),
                positive_corr_systemagent_avg_resource_dist=row.get('positive_corr_systemagent_avg_resource_dist'),
                positive_corr_systemagent_weighted_resource_dist=row.get('positive_corr_systemagent_weighted_resource_dist'),
                positive_corr_independentagent_avg_resource_dist=row.get('positive_corr_independentagent_avg_resource_dist'),
                positive_corr_independentagent_weighted_resource_dist=row.get('positive_corr_independentagent_weighted_resource_dist'),
                negative_corr_systemagent_resource_amount_in_range=row.get('negative_corr_systemagent_resource_amount_in_range'),
                negative_corr_systemagent_nearest_resource_dist=row.get('negative_corr_systemagent_nearest_resource_dist'),
                negative_corr_independentagent_resource_amount_in_range=row.get('negative_corr_independentagent_resource_amount_in_range'),
                negative_corr_controlagent_avg_resource_dist=row.get('negative_corr_controlagent_avg_resource_dist'),
                negative_corr_controlagent_nearest_resource_dist=row.get('negative_corr_controlagent_nearest_resource_dist')
            )
            session.add(resource_distribution)
            
            # Create HighLowSwitchingComparison record - focusing on key metrics
            # Note: This table has many columns, so we're only including a subset for brevity
            high_low_switching = HighLowSwitchingComparison(
                simulation_id=sim.id,
                # System agent metrics
                system_reproduction_attempts_high_switching_mean=row.get('reproduction_high_vs_low_switching_system_reproduction_attempts_high_switching_mean'),
                system_reproduction_attempts_low_switching_mean=row.get('reproduction_high_vs_low_switching_system_reproduction_attempts_low_switching_mean'),
                system_reproduction_attempts_difference=row.get('reproduction_high_vs_low_switching_system_reproduction_attempts_difference'),
                system_reproduction_attempts_percent_difference=row.get('reproduction_high_vs_low_switching_system_reproduction_attempts_percent_difference'),
                
                system_reproduction_successes_high_switching_mean=row.get('reproduction_high_vs_low_switching_system_reproduction_successes_high_switching_mean'),
                system_reproduction_successes_low_switching_mean=row.get('reproduction_high_vs_low_switching_system_reproduction_successes_low_switching_mean'),
                system_reproduction_successes_difference=row.get('reproduction_high_vs_low_switching_system_reproduction_successes_difference'),
                system_reproduction_successes_percent_difference=row.get('reproduction_high_vs_low_switching_system_reproduction_successes_percent_difference'),
                
                # Independent agent metrics
                independent_reproduction_attempts_high_switching_mean=row.get('reproduction_high_vs_low_switching_independent_reproduction_attempts_high_switching_mean'),
                independent_reproduction_attempts_low_switching_mean=row.get('reproduction_high_vs_low_switching_independent_reproduction_attempts_low_switching_mean'),
                independent_reproduction_attempts_difference=row.get('reproduction_high_vs_low_switching_independent_reproduction_attempts_difference'),
                independent_reproduction_attempts_percent_difference=row.get('reproduction_high_vs_low_switching_independent_reproduction_attempts_percent_difference'),
                
                independent_reproduction_successes_high_switching_mean=row.get('reproduction_high_vs_low_switching_independent_reproduction_successes_high_switching_mean'),
                independent_reproduction_successes_low_switching_mean=row.get('reproduction_high_vs_low_switching_independent_reproduction_successes_low_switching_mean'),
                independent_reproduction_successes_difference=row.get('reproduction_high_vs_low_switching_independent_reproduction_successes_difference'),
                independent_reproduction_successes_percent_difference=row.get('reproduction_high_vs_low_switching_independent_reproduction_successes_percent_difference'),
                
                # Control agent metrics
                control_reproduction_attempts_high_switching_mean=row.get('reproduction_high_vs_low_switching_control_reproduction_attempts_high_switching_mean'),
                control_reproduction_attempts_low_switching_mean=row.get('reproduction_high_vs_low_switching_control_reproduction_attempts_low_switching_mean'),
                control_reproduction_attempts_difference=row.get('reproduction_high_vs_low_switching_control_reproduction_attempts_difference'),
                control_reproduction_attempts_percent_difference=row.get('reproduction_high_vs_low_switching_control_reproduction_attempts_percent_difference'),
                
                control_reproduction_successes_high_switching_mean=row.get('reproduction_high_vs_low_switching_control_reproduction_successes_high_switching_mean'),
                control_reproduction_successes_low_switching_mean=row.get('reproduction_high_vs_low_switching_control_reproduction_successes_low_switching_mean'),
                control_reproduction_successes_difference=row.get('reproduction_high_vs_low_switching_control_reproduction_successes_difference'),
                control_reproduction_successes_percent_difference=row.get('reproduction_high_vs_low_switching_control_reproduction_successes_percent_difference')
            )
            session.add(high_low_switching)
            
            # Create CorrelationAnalysis record
            correlation_analysis = CorrelationAnalysis(
                simulation_id=sim.id,
                # Reproduction correlations
                repro_corr_system_reproduction_success_rate=row.get('repro_corr_system_reproduction_success_rate'),
                repro_corr_independent_avg_resources_per_reproduction=row.get('repro_corr_independent_avg_resources_per_reproduction'),
                repro_corr_independent_reproduction_success_rate=row.get('repro_corr_independent_reproduction_success_rate'),
                repro_corr_independent_reproduction_failures=row.get('repro_corr_independent_reproduction_failures'),
                repro_corr_independent_reproduction_attempts=row.get('repro_corr_independent_reproduction_attempts'),
                
                # Timing correlations
                first_reproduction_timing_correlation_system_first_reproduction_time=row.get('first_reproduction_timing_correlation_system_first_reproduction_time'),
                first_reproduction_timing_correlation_independent_first_reproduction_time=row.get('first_reproduction_timing_correlation_independent_first_reproduction_time'),
                first_reproduction_timing_correlation_control_first_reproduction_time=row.get('first_reproduction_timing_correlation_control_first_reproduction_time'),
                
                # Efficiency correlations
                reproduction_efficiency_stability_correlation_system_reproduction_efficiency=row.get('reproduction_efficiency_stability_correlation_system_reproduction_efficiency'),
                reproduction_efficiency_stability_correlation_independent_reproduction_efficiency=row.get('reproduction_efficiency_stability_correlation_independent_reproduction_efficiency'),
                reproduction_efficiency_stability_correlation_control_reproduction_efficiency=row.get('reproduction_efficiency_stability_correlation_control_reproduction_efficiency'),
                
                # System dominance correlations (sample)
                system_dominance_reproduction_correlations_system_reproduction_attempts=row.get('system_dominance_reproduction_correlations_system_reproduction_attempts'),
                system_dominance_reproduction_correlations_system_reproduction_successes=row.get('system_dominance_reproduction_correlations_system_reproduction_successes'),
                system_dominance_reproduction_correlations_system_reproduction_failures=row.get('system_dominance_reproduction_correlations_system_reproduction_failures'),
                
                # Control dominance correlations (sample)
                control_dominance_reproduction_correlations_system_reproduction_attempts=row.get('control_dominance_reproduction_correlations_system_reproduction_attempts'),
                control_dominance_reproduction_correlations_system_reproduction_successes=row.get('control_dominance_reproduction_correlations_system_reproduction_successes'),
                control_dominance_reproduction_correlations_system_reproduction_failures=row.get('control_dominance_reproduction_correlations_system_reproduction_failures')
            )
            session.add(correlation_analysis)
            
            # Commit the transaction for this simulation
            session.commit()
            
        logging.info(f"Successfully imported {len(df)} simulations into the database")
        return True
        
    except Exception as e:
        session.rollback()
        logging.error(f"Error importing data: {e}")
        return False
    finally:
        session.close()


if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "results/one_of_a_kind_50x1000/experiments/analysis/dominance/dominance_analysis.csv"
    
    # Import the data
    import_csv_to_db(csv_path) 