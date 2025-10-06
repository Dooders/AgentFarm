#!/usr/bin/env python3
"""
Shared database import utilities for analysis modules.

This module provides generic functions for importing DataFrame data into
SQLAlchemy models, supporting both single and multi-table import patterns.
"""

import logging
from typing import Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


def import_data_generic(
    df: pd.DataFrame,
    session,
    model_class,
    create_object_func: Callable,
    log_prefix: str,
    iteration_to_id: Optional[Dict[int, int]] = None,
    check_existing: bool = True,
    commit_batch_size: int = 100,
) -> int:
    """
    Generic function to import data into the database for single-table imports.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data to import
    session : SQLAlchemy session
        Database session
    model_class : SQLAlchemy model class
        The model class to check for existing records
    create_object_func : callable
        Function that takes a row and optional sim_id and returns a new model instance
    log_prefix : str
        Prefix for log messages
    iteration_to_id : dict, optional
        Mapping of iteration numbers to simulation IDs (for advantage-style imports)
    check_existing : bool
        Whether to check for existing records before importing
    commit_batch_size : int
        Number of records to commit at once

    Returns
    -------
    int
        Number of records imported
    """
    logger.info(f"Importing {log_prefix} data")
    count = 0

    for _, row in df.iterrows():
        if iteration_to_id is not None:
            iteration = int(row["iteration"])
            sim_id = iteration_to_id.get(iteration)
            if not sim_id:
                logger.warning(f"No simulation ID found for iteration {iteration}, skipping")
                continue
        else:
            sim_id = None

        # Check if data already exists
        if check_existing and sim_id is not None:
            existing = session.query(model_class).filter_by(simulation_id=sim_id).first()
            if existing:
                logger.debug(f"{log_prefix} data for simulation {iteration} already exists, skipping")
                continue

        # Create new data object
        data_object = create_object_func(row, sim_id)
        session.add(data_object)
        count += 1

        if count % commit_batch_size == 0:
            try:
                session.commit()
                logger.debug(f"Committed {count} {log_prefix} records")
            except SQLAlchemyError as e:
                logger.error(f"Error committing {log_prefix} records: {e}")
                session.rollback()
                raise

    try:
        session.commit()
        logger.info(f"Imported {count} {log_prefix} records")
    except SQLAlchemyError as e:
        logger.error(f"Error committing final {log_prefix} records: {e}")
        session.rollback()
        raise

    return count


def import_multi_table_data(
    df: pd.DataFrame,
    session,
    simulation_model_class,
    data_model_configs: List[Dict],
    log_prefix: str,
    commit_batch_size: int = 100,
) -> int:
    """
    Generic function to import data into multiple related tables per DataFrame row.

    This is designed for dominance-style imports where each row creates:
    1. One Simulation record
    2. Multiple data records that reference the simulation

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data to import
    session : SQLAlchemy session
        Database session
    simulation_model_class : SQLAlchemy model class
        The Simulation model class
    data_model_configs : list of dicts
        List of configs for data models, each containing:
        - 'model_class': The SQLAlchemy model class
        - 'create_func': Function that takes (row, sim_id) and returns model instance
        - 'name': String name for logging
    log_prefix : str
        Prefix for log messages
    commit_batch_size : int
        Number of simulation sets to commit at once

    Returns
    -------
    int
        Number of simulation records imported
    """
    logger.info(f"Importing {log_prefix} data")
    count = 0

    for _, row in df.iterrows():
        iteration = int(row["iteration"])

        # Create simulation record
        sim = simulation_model_class(iteration=iteration)
        session.add(sim)

        try:
            session.flush()  # Get the simulation ID
        except SQLAlchemyError as e:
            logger.error(f"Error creating simulation for iteration {iteration}: {e}")
            session.rollback()
            continue

        # Create all related data objects
        for config in data_model_configs:
            create_func = config.get("create_func")
            if create_func is None:
                logger.error(
                    f"Missing 'create_func' in config for {config.get('name', 'unknown')} at iteration {iteration}. Skipping this config."
                )
                continue
            try:
                data_object = create_func(row, sim.id)
                if data_object is not None:  # Allow create functions to return None to skip
                    session.add(data_object)
            except Exception as e:
                logger.error(f"Error creating {config.get('name', 'unknown')} for iteration {iteration}: {e}")
                session.rollback()
                break
        else:
            # Only increment count if all objects were created successfully
            count += 1

            if count % commit_batch_size == 0:
                try:
                    session.commit()
                    logger.debug(f"Committed {count} {log_prefix} simulation sets")
                except SQLAlchemyError as e:
                    logger.error(f"Error committing {log_prefix} records: {e}")
                    session.rollback()
                    raise

    try:
        session.commit()
        logger.info(f"Imported {count} {log_prefix} simulation sets")
    except SQLAlchemyError as e:
        logger.error(f"Error committing final {log_prefix} records: {e}")
        session.rollback()
        raise

    return count


def create_simulation_if_not_exists(
    df: pd.DataFrame, session, simulation_model_class, log_prefix: str = "simulation"
) -> Dict[int, int]:
    """
    Create simulation records for all iterations in the DataFrame if they don't exist.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing iteration data
    session : SQLAlchemy session
        Database session
    simulation_model_class : SQLAlchemy model class
        The Simulation model class
    log_prefix : str
        Prefix for log messages

    Returns
    -------
    dict
        Mapping of iteration numbers to simulation IDs
    """
    logger.info(f"Ensuring {log_prefix} records exist")
    iteration_to_id = {}

    for _, row in df.iterrows():
        iteration = int(row["iteration"])

        # Check if simulation already exists
        existing = session.query(simulation_model_class).filter_by(iteration=iteration).first()
        if existing:
            logger.debug(f"{log_prefix} {iteration} already exists, skipping")
            iteration_to_id[iteration] = existing.id
            continue

        # Create new simulation
        sim = simulation_model_class(iteration=iteration)
        session.add(sim)

        try:
            session.flush()
            iteration_to_id[iteration] = sim.id
            logger.debug(f"Added {log_prefix} {iteration} with ID {sim.id}")
        except SQLAlchemyError as e:
            logger.error(f"Error adding {log_prefix} {iteration}: {e}")
            session.rollback()

    session.commit()
    logger.info(f"Ensured {len(iteration_to_id)} {log_prefix} records exist")
    return iteration_to_id
