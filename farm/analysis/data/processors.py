"""
#! might not be used right now
Data Processing and Transformation Utilities

This module provides concrete implementations of DataProcessor classes that
transform raw data into forms suitable for analysis. Key features include:
1. Data cleaning and normalization
2. Feature extraction and engineering
3. Time series processing
4. Statistical transformations
5. Specialized processors for different analysis types

These processors implement the interfaces defined in the base module and
provide standardized data transformation capabilities.

Examples:
    # Clean data by handling missing values and outliers
    cleaner = DataCleaner(handle_missing=True, handle_outliers=True)
    cleaned_data = cleaner.process(raw_data_df)

    # Process time series data with smoothing and resampling
    ts_processor = TimeSeriesProcessor(
        resample='1H',  # Resample to hourly data
        smooth=True,    # Apply smoothing
        window_size=5   # Use 5-period rolling window
    )
    processed_ts = ts_processor.process(time_series_df)

    # Calculate agent statistics
    agent_processor = AgentStatsProcessor(include_derived_metrics=True)
    agent_stats = agent_processor.process(agents_df)

    # Perform spatial analysis on agent positions
    spatial_processor = SpatialAnalysisProcessor()
    spatial_data = spatial_processor.process(position_df)

    # Analyze resource data
    resource_processor = ResourceAnalysisProcessor()
    resource_stats = resource_processor.process(resources_df)

    # Analyze population dynamics
    pop_processor = PopulationDynamicsProcessor(
        calculate_growth_rate=True,
        calculate_stability=True
    )
    population_dynamics = pop_processor.process(population_df)

    # Calculate dominance metrics
    # Method can be 'population', 'survival', or 'reproduction'
    dominance_processor = DominanceProcessor(dominance_method='population')
    dominance_metrics = dominance_processor.process(population_df)

    # Perform feature engineering for analysis
    feature_config = {
        'is_high_energy': {
            'type': 'binary',
            'source': 'energy',
            'threshold': 50,
            'operator': '>'
        },
        'normalized_health': {
            'type': 'numeric',
            'source': 'health',
            'scaling': 'minmax'
        },
        'energy_to_health_ratio': {
            'type': 'ratio',
            'numerator': 'energy',
            'denominator': 'health'
        }
    }
    feature_processor = FeatureEngineeringProcessor(feature_config)
    features_df = feature_processor.process(agents_df)

    # Combine processors in a pipeline
    from farm.analysis.base import DataLoader, AnalysisTask
    from farm.analysis.data.loaders import SimulationLoader

    # Create a data loader
    loader = SimulationLoader(db_path='simulation.db', simulation_id=1)

    # Create processors
    cleaner = DataCleaner()
    pop_processor = PopulationDynamicsProcessor()

    # Create a task that combines the components
    task = AnalysisTask(
        data_loader=loader,
        data_processor=cleaner,
        # Additional components like analyzer, visualizer, reporter
    )

    # Run the task
    results = task.run()
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from farm.analysis.base import DataProcessor


class DataCleaner(DataProcessor):
    """Processor that cleans raw data by handling missing values, outliers, etc."""

    def __init__(self, handle_missing: bool = True, handle_outliers: bool = False):
        """Initialize the data cleaner.

        Args:
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
        """
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the input data.

        Args:
            data: Input data to clean

        Returns:
            pd.DataFrame: Cleaned data
        """
        # Make a copy to avoid modifying the original
        cleaned_data = data.copy()

        # Handle missing values
        if self.handle_missing:
            # Replace missing numeric values with the median
            numeric_cols = cleaned_data.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())

            # Replace missing categorical values with the mode
            cat_cols = cleaned_data.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in cat_cols:
                if not cleaned_data[col].empty:
                    mode_val = cleaned_data[col].mode()[0]
                    cleaned_data[col] = cleaned_data[col].fillna(mode_val)

        # Handle outliers by capping at specified percentiles
        if self.handle_outliers:
            numeric_cols = cleaned_data.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                # Get the 1st and 99th percentiles
                low = cleaned_data[col].quantile(0.01)
                high = cleaned_data[col].quantile(0.99)
                # Cap the values
                cleaned_data[col] = cleaned_data[col].clip(low, high)

        return cleaned_data


class TimeSeriesProcessor(DataProcessor):
    """Processor for time series data with operations like smoothing, resampling, etc."""

    def __init__(
        self,
        resample: Optional[str] = None,
        smooth: bool = False,
        window_size: int = 3,
        fill_missing: bool = True,
    ):
        """Initialize the time series processor.

        Args:
            resample: Resampling frequency (e.g., '1D', '1H', '5T')
            smooth: Whether to apply smoothing
            window_size: Window size for smoothing
            fill_missing: Whether to fill missing values after resampling
        """
        self.resample = resample
        self.smooth = smooth
        self.window_size = window_size
        self.fill_missing = fill_missing

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process time series data.

        Args:
            data: Input time series data

        Returns:
            pd.DataFrame: Processed time series data
        """
        # Make a copy to avoid modifying the original
        processed_data = data.copy()

        # Convert timestamp to datetime if needed
        if "timestamp" in processed_data and not pd.api.types.is_datetime64_any_dtype(
            processed_data["timestamp"]
        ):
            processed_data["timestamp"] = pd.to_datetime(processed_data["timestamp"])

        # Resample if requested
        if self.resample is not None and "timestamp" in processed_data:
            # Set timestamp as index
            processed_data = processed_data.set_index("timestamp")

            # Identify numeric columns for resampling
            numeric_cols = processed_data.select_dtypes(include=["number"]).columns

            # Resample numeric columns
            resampled = processed_data[numeric_cols].resample(self.resample).mean()

            # Fill missing values if requested
            if self.fill_missing:
                resampled = resampled.fillna(method="ffill")
                resampled = resampled.fillna(
                    method="bfill"
                )  # In case there are still NaNs at the beginning

            # Reset index to make timestamp a column again
            processed_data = resampled.reset_index()

        # Apply smoothing if requested
        if self.smooth:
            numeric_cols = processed_data.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                processed_data[col] = (
                    processed_data[col]
                    .rolling(window=self.window_size, min_periods=1)
                    .mean()
                )

        return processed_data


class AgentStatsProcessor(DataProcessor):
    """Processor for calculating agent statistics from raw agent data."""

    def __init__(self, include_derived_metrics: bool = True):
        """Initialize the agent stats processor.

        Args:
            include_derived_metrics: Whether to include derived metrics
        """
        self.include_derived_metrics = include_derived_metrics

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process agent data to calculate statistics.

        Args:
            data: Raw agent data

        Returns:
            pd.DataFrame: Processed agent statistics
        """
        # Calculate basic agent survival time
        processed_data = data.copy()
        processed_data["survival_time"] = (
            processed_data["death_step"] - processed_data["birth_step"]
        )

        # Replace NaN survival times (for still alive agents) with max step
        if "death_step" in processed_data:
            max_step = processed_data["death_step"].max()
            processed_data.loc[processed_data["death_step"].isna(), "survival_time"] = (
                max_step
                - processed_data.loc[processed_data["death_step"].isna(), "birth_step"]
            )

        # Include derived metrics if requested
        if self.include_derived_metrics:
            # Calculate survival rate by agent type
            survival_by_type = (
                processed_data.groupby("agent_type")["survival_time"]
                .agg(["mean", "median", "std", "min", "max", "count"])
                .reset_index()
            )

            # Calculate generation statistics
            if "generation" in processed_data:
                # Maximum generation reached by each agent type
                max_gen_by_type = (
                    processed_data.groupby("agent_type")["generation"]
                    .max()
                    .reset_index()
                )
                max_gen_by_type.columns = ["agent_type", "max_generation"]

                # Merge the generation stats
                survival_by_type = survival_by_type.merge(
                    max_gen_by_type, on="agent_type"
                )

                # Return the enriched data
                return survival_by_type

        return processed_data


class SpatialAnalysisProcessor(DataProcessor):
    """Processor for spatial analysis of agents and resources."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process spatial data to calculate distances and spatial metrics.

        Args:
            data: Input data with spatial coordinates

        Returns:
            pd.DataFrame: Processed spatial data
        """
        processed_data = data.copy()

        # Check if we have the necessary columns
        if not all(
            col in processed_data.columns for col in ["position_x", "position_y"]
        ):
            raise ValueError("Data must include 'position_x' and 'position_y' columns")

        # Calculate distance from origin
        processed_data["distance_from_origin"] = np.sqrt(
            processed_data["position_x"] ** 2 + processed_data["position_y"] ** 2
        )

        # Calculate nearest neighbor distances (if there are multiple points)
        if len(processed_data) > 1:
            # This is a simple implementation - for large datasets, consider using a KDTree
            nearest_distances = []
            points = processed_data[["position_x", "position_y"]].values

            for i, point in enumerate(points):
                # Calculate distances to all other points
                distances = [
                    euclidean(point, other_point)
                    for j, other_point in enumerate(points)
                    if i != j
                ]
                # Find the minimum distance
                nearest_distances.append(min(distances) if distances else np.nan)

            processed_data["nearest_neighbor_distance"] = nearest_distances

        return processed_data


class ResourceAnalysisProcessor(DataProcessor):
    """Processor for analyzing resource data."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process resource data.

        Args:
            data: Input resource data

        Returns:
            pd.DataFrame: Processed resource data
        """
        processed_data = data.copy()

        # Calculate resource lifespan
        if all(
            col in processed_data.columns for col in ["creation_step", "depletion_step"]
        ):
            # Filter out resources that haven't been depleted yet
            depleted = processed_data.dropna(subset=["depletion_step"])
            depleted["lifespan"] = (
                depleted["depletion_step"] - depleted["creation_step"]
            )

            # Group by resource type
            resource_stats = (
                depleted.groupby("resource_type")
                .agg(
                    mean_lifespan=("lifespan", "mean"),
                    median_lifespan=("lifespan", "median"),
                    min_lifespan=("lifespan", "min"),
                    max_lifespan=("lifespan", "max"),
                    count=("id", "count"),
                )
                .reset_index()
            )

            return resource_stats

        return processed_data


class PopulationDynamicsProcessor(DataProcessor):
    """Processor for analyzing population dynamics from time series data."""

    def __init__(
        self, calculate_growth_rate: bool = True, calculate_stability: bool = True
    ):
        """Initialize the population dynamics processor.

        Args:
            calculate_growth_rate: Whether to calculate growth rates
            calculate_stability: Whether to calculate population stability metrics
        """
        self.calculate_growth_rate = calculate_growth_rate
        self.calculate_stability = calculate_stability

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process population time series data.

        Args:
            data: Input population time series data
                Expected format: DataFrame with 'step', 'agent_type', 'count' columns

        Returns:
            pd.DataFrame: Processed population dynamics data
        """
        # Check if we have the necessary columns
        required_cols = ["step", "agent_type", "count"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must include columns: {required_cols}")

        # Pivot the data to have agent types as columns
        pivoted = data.pivot(
            index="step", columns="agent_type", values="count"
        ).reset_index()
        pivoted = pivoted.fillna(0)

        # Calculate growth rates if requested
        if self.calculate_growth_rate:
            agent_types = [col for col in pivoted.columns if col != "step"]
            for agent_type in agent_types:
                # Calculate the period-to-period growth rate
                pivoted[f"{agent_type}_growth_rate"] = pivoted[agent_type].pct_change()

                # Calculate the cumulative growth rate
                pivoted[f"{agent_type}_cumulative_growth"] = (
                    pivoted[agent_type] / pivoted[agent_type].iloc[0]
                    if pivoted[agent_type].iloc[0] > 0
                    else 0
                )

        # Calculate stability metrics if requested
        if self.calculate_stability:
            agent_types = [
                col
                for col in pivoted.columns
                if col != "step"
                and not col.endswith("_growth_rate")
                and not col.endswith("_cumulative_growth")
            ]
            for agent_type in agent_types:
                # Calculate the rolling standard deviation (10-period window by default)
                pivoted[f"{agent_type}_volatility"] = (
                    pivoted[agent_type].rolling(window=10, min_periods=1).std()
                )

                # Calculate the coefficient of variation (normalized volatility)
                rolling_mean = (
                    pivoted[agent_type].rolling(window=10, min_periods=1).mean()
                )
                pivoted[f"{agent_type}_cv"] = (
                    pivoted[f"{agent_type}_volatility"] / rolling_mean
                )
                pivoted[f"{agent_type}_cv"] = pivoted[f"{agent_type}_cv"].fillna(0)

        return pivoted


class DominanceProcessor(DataProcessor):
    """Processor for calculating dominance metrics."""

    def __init__(self, dominance_method: str = "population"):
        """Initialize the dominance processor.

        Args:
            dominance_method: Method for calculating dominance
                'population': Based on final population
                'survival': Based on survival time
                'reproduction': Based on reproduction events
        """
        self.dominance_method = dominance_method

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data to calculate dominance metrics.

        Args:
            data: Input data for dominance calculation

        Returns:
            pd.DataFrame: Processed dominance data
        """
        # The processing depends on the dominance method
        if self.dominance_method == "population":
            return self._process_population_dominance(data)
        elif self.dominance_method == "survival":
            return self._process_survival_dominance(data)
        elif self.dominance_method == "reproduction":
            return self._process_reproduction_dominance(data)
        else:
            raise ValueError(f"Unknown dominance method: {self.dominance_method}")

    def _process_population_dominance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate dominance based on population counts.

        Args:
            data: Population data (expected to have 'step', 'agent_type', 'count' columns)

        Returns:
            pd.DataFrame: Dominance metrics
        """
        # Check if we have the necessary columns
        required_cols = ["step", "agent_type", "count"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must include columns: {required_cols}")

        # Find the final step for each simulation
        if "simulation_id" in data.columns:
            final_steps = data.groupby("simulation_id")["step"].max().reset_index()
            final_data = pd.merge(data, final_steps, on=["simulation_id", "step"])
        else:
            final_step = data["step"].max()
            final_data = data[data["step"] == final_step]

        # Calculate dominance metrics
        dominance_metrics = []

        if "simulation_id" in data.columns:
            for sim_id, sim_data in final_data.groupby("simulation_id"):
                total_agents = sim_data["count"].sum()
                if total_agents > 0:
                    for _, row in sim_data.iterrows():
                        dominance_metrics.append(
                            {
                                "simulation_id": sim_id,
                                "agent_type": row["agent_type"],
                                "final_count": row["count"],
                                "dominance_score": row["count"] / total_agents,
                                "is_dominant": row["count"] == sim_data["count"].max(),
                            }
                        )
        else:
            total_agents = final_data["count"].sum()
            if total_agents > 0:
                for _, row in final_data.iterrows():
                    dominance_metrics.append(
                        {
                            "agent_type": row["agent_type"],
                            "final_count": row["count"],
                            "dominance_score": row["count"] / total_agents,
                            "is_dominant": row["count"] == final_data["count"].max(),
                        }
                    )

        return pd.DataFrame(dominance_metrics)

    def _process_survival_dominance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate dominance based on survival time.

        Args:
            data: Agent data (expected to have 'agent_type', 'birth_step', 'death_step' columns)

        Returns:
            pd.DataFrame: Dominance metrics
        """
        # Check if we have the necessary columns
        required_cols = ["agent_type", "birth_step"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must include columns: {required_cols}")

        # Calculate survival time
        processed_data = data.copy()

        if "death_step" in processed_data.columns:
            max_step = processed_data["death_step"].max()
            processed_data["survival_time"] = (
                processed_data["death_step"] - processed_data["birth_step"]
            )
            # For agents still alive, use the max step
            processed_data.loc[processed_data["death_step"].isna(), "survival_time"] = (
                max_step
                - processed_data.loc[processed_data["death_step"].isna(), "birth_step"]
            )
        else:
            max_step = processed_data["birth_step"].max()
            processed_data["survival_time"] = max_step - processed_data["birth_step"]

        # Calculate dominance metrics by agent type
        dominance_metrics = []

        if "simulation_id" in processed_data.columns:
            for sim_id, sim_data in processed_data.groupby("simulation_id"):
                for agent_type, type_data in sim_data.groupby("agent_type"):
                    dominance_metrics.append(
                        {
                            "simulation_id": sim_id,
                            "agent_type": agent_type,
                            "mean_survival": type_data["survival_time"].mean(),
                            "max_survival": type_data["survival_time"].max(),
                            "agent_count": len(type_data),
                        }
                    )
        else:
            for agent_type, type_data in processed_data.groupby("agent_type"):
                dominance_metrics.append(
                    {
                        "agent_type": agent_type,
                        "mean_survival": type_data["survival_time"].mean(),
                        "max_survival": type_data["survival_time"].max(),
                        "agent_count": len(type_data),
                    }
                )

        # Convert to DataFrame
        dominance_df = pd.DataFrame(dominance_metrics)

        # Calculate dominance score and is_dominant
        if not dominance_df.empty:
            if "simulation_id" in dominance_df.columns:
                for sim_id, sim_data in dominance_df.groupby("simulation_id"):
                    max_survival = sim_data["mean_survival"].max()
                    dominance_df.loc[
                        dominance_df["simulation_id"] == sim_id, "dominance_score"
                    ] = (
                        dominance_df.loc[
                            dominance_df["simulation_id"] == sim_id, "mean_survival"
                        ]
                        / max_survival
                    )
                    dominance_df.loc[
                        dominance_df["simulation_id"] == sim_id, "is_dominant"
                    ] = (
                        dominance_df.loc[
                            dominance_df["simulation_id"] == sim_id, "mean_survival"
                        ]
                        == max_survival
                    )
            else:
                max_survival = dominance_df["mean_survival"].max()
                dominance_df["dominance_score"] = (
                    dominance_df["mean_survival"] / max_survival
                )
                dominance_df["is_dominant"] = (
                    dominance_df["mean_survival"] == max_survival
                )

        return dominance_df

    def _process_reproduction_dominance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate dominance based on reproduction events.

        Args:
            data: Reproduction data

        Returns:
            pd.DataFrame: Dominance metrics
        """
        # For reproduction dominance, we need to join with agent data
        # to get the agent type for each parent

        # Check if we have parent_id
        if "parent_id" not in data.columns:
            raise ValueError(
                "Data must include 'parent_id' column for reproduction dominance"
            )

        # If we already have agent_type, use it directly
        if "agent_type" in data.columns:
            reproduction_counts = (
                data.groupby("agent_type").size().reset_index(name="reproduction_count")
            )

            # Calculate dominance metrics
            total_reproductions = reproduction_counts["reproduction_count"].sum()
            if total_reproductions > 0:
                reproduction_counts["dominance_score"] = (
                    reproduction_counts["reproduction_count"] / total_reproductions
                )
                reproduction_counts["is_dominant"] = (
                    reproduction_counts["reproduction_count"]
                    == reproduction_counts["reproduction_count"].max()
                )
            else:
                reproduction_counts["dominance_score"] = 0
                reproduction_counts["is_dominant"] = False

            return reproduction_counts
        else:
            # Without agent_type, we can't calculate reproduction dominance
            raise ValueError(
                "Data must include 'agent_type' column for reproduction dominance"
            )


class FeatureEngineeringProcessor(DataProcessor):
    """Processor for feature engineering and transformation."""

    def __init__(self, feature_config: Dict[str, Dict[str, Any]]):
        """Initialize the feature engineering processor.

        Args:
            feature_config: Configuration for feature engineering
                Dictionary mapping feature names to transformation configs
        """
        self.feature_config = feature_config

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features based on the feature configuration.

        Args:
            data: Input data

        Returns:
            pd.DataFrame: Data with engineered features
        """
        processed_data = data.copy()

        for feature_name, config in self.feature_config.items():
            feature_type = config.get("type", "identity")

            if feature_type == "identity":
                # Copy an existing column as a new feature
                source_col = config.get("source")
                if source_col in processed_data.columns:
                    processed_data[feature_name] = processed_data[source_col]

            elif feature_type == "binary":
                # Create a binary feature based on a condition
                source_col = config.get("source")
                threshold = config.get("threshold", 0)
                operator = config.get("operator", ">")

                if source_col in processed_data.columns:
                    if operator == ">":
                        processed_data[feature_name] = (
                            processed_data[source_col] > threshold
                        ).astype(int)
                    elif operator == ">=":
                        processed_data[feature_name] = (
                            processed_data[source_col] >= threshold
                        ).astype(int)
                    elif operator == "<":
                        processed_data[feature_name] = (
                            processed_data[source_col] < threshold
                        ).astype(int)
                    elif operator == "<=":
                        processed_data[feature_name] = (
                            processed_data[source_col] <= threshold
                        ).astype(int)
                    elif operator == "==":
                        processed_data[feature_name] = (
                            processed_data[source_col] == threshold
                        ).astype(int)
                    elif operator == "!=":
                        processed_data[feature_name] = (
                            processed_data[source_col] != threshold
                        ).astype(int)

            elif feature_type == "categorical":
                # Convert a column to categorical and optionally one-hot encode
                source_col = config.get("source")
                one_hot = config.get("one_hot", False)

                if source_col in processed_data.columns:
                    if one_hot:
                        # Create one-hot encoded columns
                        one_hot_cols = pd.get_dummies(
                            processed_data[source_col], prefix=feature_name
                        )
                        processed_data = pd.concat(
                            [processed_data, one_hot_cols], axis=1
                        )
                    else:
                        # Just convert to category
                        processed_data[feature_name] = processed_data[
                            source_col
                        ].astype("category")

            elif feature_type == "numeric":
                # Create a numeric feature, optionally scaled
                source_col = config.get("source")
                scaling = config.get("scaling", None)

                if source_col in processed_data.columns:
                    if scaling == "minmax":
                        scaler = MinMaxScaler()
                        processed_data[feature_name] = scaler.fit_transform(
                            processed_data[source_col].values.reshape(-1, 1)
                        )
                    elif scaling == "standard":
                        scaler = StandardScaler()
                        processed_data[feature_name] = scaler.fit_transform(
                            processed_data[source_col].values.reshape(-1, 1)
                        )
                    else:
                        processed_data[feature_name] = processed_data[source_col]

            elif feature_type == "ratio":
                # Calculate a ratio between two columns
                numerator = config.get("numerator")
                denominator = config.get("denominator")
                default_value = config.get("default", 0)

                if (
                    numerator in processed_data.columns
                    and denominator in processed_data.columns
                ):
                    processed_data[feature_name] = (
                        processed_data[numerator] / processed_data[denominator]
                    )
                    processed_data[feature_name] = processed_data[feature_name].fillna(
                        default_value
                    )
                    processed_data[feature_name] = processed_data[feature_name].replace(
                        [np.inf, -np.inf], default_value
                    )

            elif feature_type == "interaction":
                # Create an interaction feature (product of two columns)
                col1 = config.get("col1")
                col2 = config.get("col2")

                if col1 in processed_data.columns and col2 in processed_data.columns:
                    processed_data[feature_name] = (
                        processed_data[col1] * processed_data[col2]
                    )

            elif feature_type == "aggregate":
                # Create an aggregate feature (e.g., rolling mean)
                source_col = config.get("source")
                groupby_col = config.get("groupby")
                agg_func = config.get("function", "mean")

                if (
                    source_col in processed_data.columns
                    and groupby_col in processed_data.columns
                ):
                    agg_values = processed_data.groupby(groupby_col)[
                        source_col
                    ].transform(agg_func)
                    processed_data[feature_name] = agg_values

        return processed_data
