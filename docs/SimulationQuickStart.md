# Quickstart Guide: From Repository to Simulation Results

Welcome to AgentFarm, a digital research platform for simulations of complex systems. This guide will help you quickly set up and run simulations to get your results.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.8 or higher
- `pip` (Python package installer)
- Git

## Step 1: Clone the Repository

First, clone the AgentFarm repository to your local machine using Git.

```bash
git clone https://github.com/Dooders/AgentFarm.git
cd AgentFarm
```

## Step 2: Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Create and activate a virtual environment using the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Step 3: Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Step 4: Configure Your Simulation

Navigate to the `config` directory and create a configuration file for your simulation. You can use the provided template `config_template.yaml` as a starting point.

```bash
cd config
cp config_template.yaml my_simulation_config.yaml
```

Edit `my_simulation_config.yaml` to set up your simulation parameters.

## Step 5: Run the Simulation

With your configuration file ready, you can now run the simulation. Navigate back to the root directory and execute the simulation script:

```bash
cd ..
python run_simulation.py --config config/my_simulation_config.yaml
```

## Step 6: View the Results

Once the simulation completes, results will be saved in the `results` directory. You can analyze the results using your preferred data analysis tools.

## Additional Information

- For more detailed instructions and advanced configurations, refer to the [documentation](docs/README.md).
- If you encounter any issues, please check the [issues page](https://github.com/Dooders/AgentFarm/issues) or open a new issue for assistance.

---

Congratulations! You have successfully set up and run a simulation using AgentFarm. Happy simulating!
