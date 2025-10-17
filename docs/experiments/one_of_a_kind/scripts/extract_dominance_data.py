import json
import os
import sqlite3
import sys


def extract_dominance_data(db_path):
    """
    Extract key dominance metrics from the database and return as a structured dictionary.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        data = {
            "dominance_distribution": {
                "population_dominance": {},
                "survival_dominance": {},
                "comprehensive_dominance": {},
            },
            "dominance_switching": {
                "average_switches_per_simulation": None,
                "average_switches_per_step": None,
                "switches_by_phase": {"early": None, "middle": None, "late": None},
            },
            "dominance_period_duration": {},
            "reproduction_strategy": {
                "stability_coefficients": {},
                "switches_preceded_by_reproduction_changes_percent": None,
            },
        }

        # Query for dominance distribution
        cursor.execute(
            """
            SELECT population_dominance, COUNT(*) as count
            FROM dominance_metrics
            GROUP BY population_dominance
        """
        )

        total_count = 0
        population_counts = {}
        for agent_type, count in cursor.fetchall():
            population_counts[agent_type] = count
            total_count += count

        for agent_type, count in population_counts.items():
            data["dominance_distribution"]["population_dominance"][agent_type] = round(
                count / total_count * 100, 1
            )

        # Query for survival dominance
        cursor.execute(
            """
            SELECT survival_dominance, COUNT(*) as count
            FROM dominance_metrics
            GROUP BY survival_dominance
        """
        )

        total_count = 0
        survival_counts = {}
        for agent_type, count in cursor.fetchall():
            survival_counts[agent_type] = count
            total_count += count

        for agent_type, count in survival_counts.items():
            data["dominance_distribution"]["survival_dominance"][agent_type] = round(
                count / total_count * 100, 1
            )

        # Query for comprehensive dominance
        cursor.execute(
            """
            SELECT comprehensive_dominance, COUNT(*) as count
            FROM dominance_metrics
            GROUP BY comprehensive_dominance
        """
        )

        total_count = 0
        comprehensive_counts = {}
        for agent_type, count in cursor.fetchall():
            comprehensive_counts[agent_type] = count
            total_count += count

        for agent_type, count in comprehensive_counts.items():
            data["dominance_distribution"]["comprehensive_dominance"][agent_type] = (
                round(count / total_count * 100, 1)
            )

        # Query for dominance switching metrics
        cursor.execute(
            """
            SELECT AVG(total_switches) as avg_switches,
                   AVG(switches_per_step) as switches_per_step,
                   AVG(early_phase_switches) as early_switches,
                   AVG(middle_phase_switches) as middle_switches,
                   AVG(late_phase_switches) as late_switches
            FROM dominance_switching
        """
        )

        row = cursor.fetchone()
        if row:
            data["dominance_switching"]["average_switches_per_simulation"] = round(
                row[0], 1
            )
            data["dominance_switching"]["average_switches_per_step"] = round(row[1], 4)
            data["dominance_switching"]["switches_by_phase"]["early"] = round(row[2], 1)
            data["dominance_switching"]["switches_by_phase"]["middle"] = round(
                row[3], 1
            )
            data["dominance_switching"]["switches_by_phase"]["late"] = round(row[4], 1)

        # Query for dominance period duration
        cursor.execute(
            """
            SELECT AVG(system_avg_dominance_period) as system_period,
                   AVG(independent_avg_dominance_period) as independent_period,
                   AVG(control_avg_dominance_period) as control_period
            FROM dominance_switching
        """
        )

        row = cursor.fetchone()
        if row:
            data["dominance_period_duration"]["System"] = round(row[0], 1)
            data["dominance_period_duration"]["Independent"] = round(row[1], 1)
            data["dominance_period_duration"]["Control"] = round(row[2], 1)

        conn.close()
        return data

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level to one_of_a_kind directory
    data_dir = os.path.join(base_dir, "data")  # Path to data directory

    # Set default database path in data directory
    db_path = os.path.join(data_dir, "dominance.db")

    # Override default path if provided as argument
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    data = extract_dominance_data(db_path)

    if data:
        print(json.dumps(data, indent=2))

        # Save to JSON file in data directory
        output_path = os.path.join(data_dir, "dominance_metrics.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {output_path}")
    else:
        print("Failed to extract data from the database.")


if __name__ == "__main__":
    main()
