import pandas as pd
import matplotlib.pyplot as plt

# Define the analysis functions

def plot_resource_amount_distribution(dataframe):
    """Plot the distribution of resource amounts."""
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe['amount'], bins=30, edgecolor='k', alpha=0.7, color='green')
    plt.title('Resource Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.show()

def plot_resource_positions(dataframe):
    """Plot the spatial distribution of resources."""
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe['position_x'], dataframe['position_y'], c=dataframe['amount'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Resource Amount')
    plt.title('Resource Positions and Amounts')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def plot_total_resources_over_time(dataframe):
    """Plot the total amount of resources over time."""
    total_resources = dataframe.groupby('step_number')['amount'].sum()
    plt.figure(figsize=(10, 6))
    plt.plot(total_resources.index, total_resources.values, marker='o', label='Total Resources')
    plt.title('Total Resources Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Total Resource Amount')
    plt.legend()
    plt.show()

def plot_resource_distribution_by_step(dataframe, step_number):
    """Plot the spatial distribution of resources for a specific step."""
    step_data = dataframe[dataframe['step_number'] == step_number]
    plt.figure(figsize=(10, 6))
    plt.scatter(step_data['position_x'], step_data['position_y'], c=step_data['amount'], cmap='plasma', alpha=0.7)
    plt.colorbar(label='Resource Amount')
    plt.title(f'Resource Distribution at Step {step_number}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def plot_resource_amount_changes(dataframe, resource_id):
    """Plot the amount of a specific resource over time."""
    resource_data = dataframe[dataframe['resource_id'] == resource_id]
    plt.figure(figsize=(10, 6))
    plt.plot(resource_data['step_number'], resource_data['amount'], marker='o', label=f'Resource {resource_id}')
    plt.title(f'Resource Amount Over Time for Resource {resource_id}')
    plt.xlabel('Step Number')
    plt.ylabel('Amount')
    plt.legend()
    plt.show()

def plot_average_resource_density(dataframe, grid_size):
    """Plot average resource density based on a grid."""
    dataframe['x_bin'] = (dataframe['position_x'] // grid_size).astype(int)
    dataframe['y_bin'] = (dataframe['position_y'] // grid_size).astype(int)
    density = dataframe.groupby(['x_bin', 'y_bin'])['amount'].mean().unstack(fill_value=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(density, cmap='YlGnBu', origin='lower', interpolation='nearest')
    plt.colorbar(label='Average Resource Amount')
    plt.title(f'Average Resource Density (Grid Size {grid_size})')
    plt.xlabel('X Bins')
    plt.ylabel('Y Bins')
    plt.show()

# Load the dataset
def main(dataframe):

    try:

        # Call each function to analyze and visualize
        step_number = 10  # Specify the step number for analysis
        resource_id = 1  # Specify the resource ID for analysis
        grid_size = 10  # Specify the grid size for density analysis

        print("Plotting resource amount distribution...")
        plot_resource_amount_distribution(dataframe)

        print("Plotting resource positions and amounts...")
        plot_resource_positions(dataframe)

        print("Plotting total resources over time...")
        plot_total_resources_over_time(dataframe)

        print(f"Plotting resource distribution at step {step_number}...")
        plot_resource_distribution_by_step(dataframe, step_number)

        print(f"Plotting resource amount changes for resource {resource_id}...")
        plot_resource_amount_changes(dataframe, resource_id)

        print(f"Plotting average resource density with grid size {grid_size}...")
        plot_average_resource_density(dataframe, grid_size)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the analysis
if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine, inspect

    # connection_string = "sqlite:///simulations/simulation_20241110_122335.db"
    connection_string = "sqlite:///simulations/simulation.db"

    # Create engine
    engine = create_engine(connection_string)
    
    df = pd.read_sql("SELECT * FROM Resource_States", engine)

    main(df)
