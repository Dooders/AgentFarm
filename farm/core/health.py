"""Biomimetic health system implementation for AgentFarm agents.

This module implements a sophisticated health system modeled after biological systems.
It includes homeostasis, allostatic load tracking, multi-dimensional health vectors,
and environmental adaptation mechanisms.

Key components:
- HealthVector: Multi-dimensional health representation
- VitalParameters: Homeostatic regulation of physical parameters
- StressResponse: Allostatic load and stress hormone system
- RecoverySystem: Resource-dependent healing mechanics
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class HealthDimension(Enum):
    """Represents core health dimensions in the biomimetic health system."""
    PHYSICAL = "physical_integrity"
    METABOLIC = "metabolic_efficiency"
    IMMUNE = "immune_function"
    NEURAL = "neural_health"


class VitalParameter(Enum):
    """Physiological parameters that must be maintained within ranges."""
    TEMPERATURE = "temperature"  # Body temperature (°C)
    PH = "ph_balance"  # Blood/tissue pH level
    GLUCOSE = "glucose"  # Blood sugar level
    OSMOTIC = "osmotic_pressure"  # Fluid balance


@dataclass
class ParameterRange:
    """Defines optimal and tolerable ranges for vital parameters."""
    optimal_min: float
    optimal_max: float
    critical_min: float
    critical_max: float
    
    def is_optimal(self, value: float) -> bool:
        """Check if value is within optimal range."""
        return self.optimal_min <= value <= self.optimal_max
    
    def is_critical(self, value: float) -> bool:
        """Check if value is outside critical range."""
        return value < self.critical_min or value > self.critical_max
    
    def get_deviation(self, value: float) -> float:
        """Calculate deviation from optimal range.
        
        Returns 0.0 if within optimal range, otherwise the distance to
        the nearest optimal boundary normalized by the distance between
        optimal and critical boundaries.
        """
        if self.is_optimal(value):
            return 0.0
            
        if value < self.optimal_min:
            return (self.optimal_min - value) / (self.optimal_min - self.critical_min)
        else:
            return (value - self.optimal_max) / (self.critical_max - self.optimal_max)


@dataclass
class HealthVector:
    """Multi-dimensional health representation."""
    physical_integrity: float = 100.0  # Physical structure/damage
    metabolic_efficiency: float = 100.0  # Energy processing
    immune_function: float = 100.0  # Resistance to threats
    neural_health: float = 100.0  # Decision-making capability
    
    # Weights used when calculating overall health
    dimension_weights: Dict[HealthDimension, float] = field(default_factory=lambda: {
        HealthDimension.PHYSICAL: 0.4,
        HealthDimension.METABOLIC: 0.3,
        HealthDimension.IMMUNE: 0.2,
        HealthDimension.NEURAL: 0.1
    })
    
    def get_overall_health(self) -> float:
        """Calculate weighted overall health value."""
        total = 0.0
        for dim, weight in self.dimension_weights.items():
            value = getattr(self, dim.value)
            total += value * weight
        return total
    
    def apply_damage(self, dimension: HealthDimension, amount: float) -> float:
        """Apply damage to a specific health dimension.
        
        Returns:
            float: Amount of damage actually applied
        """
        current = getattr(self, dimension.value)
        new_value = max(0.0, current - amount)
        setattr(self, dimension.value, new_value)
        return current - new_value
    
    def heal(self, dimension: HealthDimension, amount: float) -> float:
        """Heal a specific health dimension.
        
        Returns:
            float: Amount of healing actually applied
        """
        current = getattr(self, dimension.value)
        new_value = min(100.0, current + amount)
        setattr(self, dimension.value, new_value)
        return new_value - current
    
    def to_dict(self) -> Dict[str, float]:
        """Convert health vector to dictionary."""
        return {
            "physical_integrity": self.physical_integrity,
            "metabolic_efficiency": self.metabolic_efficiency,
            "immune_function": self.immune_function,
            "neural_health": self.neural_health,
            "overall_health": self.get_overall_health()
        }


@dataclass
class HomeostasisSystem:
    """System that regulates vital parameters within optimal ranges."""
    
    # Default parameter ranges (can be customized per agent/species)
    parameter_ranges: Dict[VitalParameter, ParameterRange] = field(default_factory=lambda: {
        VitalParameter.TEMPERATURE: ParameterRange(36.5, 37.5, 35.0, 39.0),
        VitalParameter.PH: ParameterRange(7.35, 7.45, 7.0, 7.8),
        VitalParameter.GLUCOSE: ParameterRange(3.9, 5.5, 2.8, 7.8),
        VitalParameter.OSMOTIC: ParameterRange(280, 295, 265, 320)
    })
    
    # Current parameter values
    current_values: Dict[VitalParameter, float] = field(default_factory=lambda: {
        VitalParameter.TEMPERATURE: 37.0,
        VitalParameter.PH: 7.4,
        VitalParameter.GLUCOSE: 4.5,
        VitalParameter.OSMOTIC: 285
    })
    
    # Regulatory efficiency (0-1) - how well the agent can maintain homeostasis
    regulation_efficiency: Dict[VitalParameter, float] = field(default_factory=lambda: {
        VitalParameter.TEMPERATURE: 0.8,
        VitalParameter.PH: 0.7,
        VitalParameter.GLUCOSE: 0.6,
        VitalParameter.OSMOTIC: 0.7
    })
    
    def update_temperature(self, environmental_temp: float) -> Tuple[float, float]:
        """Update body temperature based on environmental temperature.
        
        Args:
            environmental_temp: Current environmental temperature
            
        Returns:
            Tuple containing:
                - Current temperature value
                - Homeostatic stress (0-1 scale, 0 = no stress)
        """
        # Get current body temperature
        body_temp = self.current_values[VitalParameter.TEMPERATURE]
        
        # Calculate temperature differential
        temp_diff = environmental_temp - body_temp
        
        # Apply temperature change modified by regulation efficiency
        reg_efficiency = self.regulation_efficiency[VitalParameter.TEMPERATURE]
        new_temp = body_temp + temp_diff * (1 - reg_efficiency) * 0.1
        
        # Update stored value
        self.current_values[VitalParameter.TEMPERATURE] = new_temp
        
        # Calculate homeostatic stress based on deviation from optimal
        temp_range = self.parameter_ranges[VitalParameter.TEMPERATURE]
        stress = temp_range.get_deviation(new_temp)
        
        return new_temp, stress
    
    def get_stress_level(self) -> float:
        """Calculate overall homeostatic stress across all parameters.
        
        Returns:
            float: Stress level from 0 (no stress) to 1 (maximum stress)
        """
        stress = 0.0
        for param, value in self.current_values.items():
            param_range = self.parameter_ranges[param]
            stress += param_range.get_deviation(value)
        
        # Normalize by number of parameters and cap at 1.0
        return min(1.0, stress / len(self.current_values))


@dataclass
class AllostaticSystem:
    """Tracks cumulative stress and adaptation to stressors."""
    
    # Current stress hormone levels
    stress_hormones: float = 0.0  # 0-100 scale
    
    # Accumulated inflammation markers
    inflammation: float = 0.0  # 0-100 scale
    
    # Oxidative stress damage
    oxidative_stress: float = 0.0  # 0-100 scale
    
    # Recovery capacity (decreases with age/chronic stress)
    recovery_capacity: float = 1.0  # 0-1 scale
    
    # Stress exposure history (recent stressors)
    recent_stressors: List[Tuple[str, float]] = field(default_factory=list)
    
    def add_stress(self, stress_type: str, intensity: float) -> None:
        """Add a stress event to the system.
        
        Args:
            stress_type: Type of stressor (e.g., "temperature", "combat")
            intensity: Intensity of stress (0-1 scale)
        """
        # Increase stress hormone levels (with diminishing returns)
        hormone_increase = intensity * (1.0 - self.stress_hormones / 100.0) * 20.0
        self.stress_hormones = min(100.0, self.stress_hormones + hormone_increase)
        
        # Increase inflammation (more for repeated stressors)
        stressor_count = sum(1 for s in self.recent_stressors if s[0] == stress_type)
        inflammation_increase = intensity * (1.0 + stressor_count * 0.2) * 5.0
        self.inflammation = min(100.0, self.inflammation + inflammation_increase)
        
        # Increase oxidative stress
        oxidative_increase = intensity * 2.0
        self.oxidative_stress = min(100.0, self.oxidative_stress + oxidative_increase)
        
        # Record stressor (keep last 10)
        self.recent_stressors.append((stress_type, intensity))
        if len(self.recent_stressors) > 10:
            self.recent_stressors.pop(0)
    
    def recover(self, rest_intensity: float = 0.5) -> None:
        """Recover from stress based on rest and recovery capacity.
        
        Args:
            rest_intensity: How effective the rest is (0-1 scale)
        """
        recovery_factor = rest_intensity * self.recovery_capacity
        
        # Decrease stress hormones
        self.stress_hormones = max(0.0, self.stress_hormones - recovery_factor * 10.0)
        
        # Decrease inflammation (slower)
        self.inflammation = max(0.0, self.inflammation - recovery_factor * 3.0)
        
        # Decrease oxidative stress (slowest)
        self.oxidative_stress = max(0.0, self.oxidative_stress - recovery_factor * 1.0)
    
    def get_allostatic_load(self) -> float:
        """Calculate current allostatic load (cumulative stress burden).
        
        Returns:
            float: Allostatic load from 0 (none) to 100 (maximum)
        """
        return (self.stress_hormones * 0.3 + 
                self.inflammation * 0.4 + 
                self.oxidative_stress * 0.3)


class BiomimeticHealthSystem:
    """Complete biomimetic health system for an agent."""
    
    def __init__(self, starting_health: float = 100.0):
        """Initialize the health system.
        
        Args:
            starting_health: Initial overall health value
        """
        # Initialize health vector with starting values
        self.health_vector = HealthVector(
            physical_integrity=starting_health,
            metabolic_efficiency=starting_health,
            immune_function=starting_health,
            neural_health=starting_health
        )
        
        # Initialize homeostasis system
        self.homeostasis = HomeostasisSystem()
        
        # Initialize allostatic system
        self.allostatic_system = AllostaticSystem()
        
        # Resource stockpiles for healing
        self.resource_stockpiles = {
            "proteins": 0.0,
            "carbohydrates": 0.0,
            "lipids": 0.0,
            "minerals": 0.0
        }
        
        # Temperature adaptation
        self.adapted_temperature = 20.0  # Starting adapted temperature (°C)
        self.adaptation_rate = 0.01  # How quickly agent adapts to temperatures
        
        # Aging system
        self.age = 0
        self.cellular_divisions = 0
        self.hayflick_limit = 50  # Maximum number of divisions
        self.non_repairable_damage = 0.0
        
        # Behavior modifiers based on health state
        self.behavior_modifiers = {
            "movement_speed": 1.0,
            "attack_strength": 1.0,
            "gather_efficiency": 1.0,
            "decision_quality": 1.0
        }
    
    def update(self, environmental_temperature: float, is_resting: bool = False) -> Dict:
        """Update health state based on environmental conditions.
        
        Args:
            environmental_temperature: Current environmental temperature
            is_resting: Whether agent is in a rest state
            
        Returns:
            Dict containing updated health metrics
        """
        # Update body temperature based on environment
        body_temp, temp_stress = self.homeostasis.update_temperature(environmental_temperature)
        
        # Update temperature adaptation
        if abs(self.adapted_temperature - environmental_temperature) > 1.0:
            self.adapted_temperature += (environmental_temperature - self.adapted_temperature) * self.adaptation_rate
        
        # Calculate temperature stress based on adaptation
        adapted_temp_stress = temp_stress * (1.0 - 0.5 * (1.0 / (1.0 + abs(environmental_temperature - self.adapted_temperature))))
        
        # Add stress to allostatic system if significant
        if adapted_temp_stress > 0.1:
            self.allostatic_system.add_stress("temperature", adapted_temp_stress)
        
        # Apply recovery if resting
        if is_resting:
            self.allostatic_system.recover(0.8)
        else:
            self.allostatic_system.recover(0.2)  # Minimal recovery when active
        
        # Apply allostatic load effects to health vector
        allostatic_load = self.allostatic_system.get_allostatic_load()
        if allostatic_load > 30:  # Only damage if load is significant
            load_damage = (allostatic_load - 30) / 70.0 * 0.5  # Max 0.5 damage per update
            self.health_vector.apply_damage(HealthDimension.METABOLIC, load_damage)
            self.health_vector.apply_damage(HealthDimension.IMMUNE, load_damage * 0.8)
        
        # Apply temperature-specific effects
        if temp_stress > 0.5:  # Significant temperature deviation
            if body_temp > 38.0:  # High temperature
                # High temps primarily affect metabolic and neural systems
                self.health_vector.apply_damage(HealthDimension.METABOLIC, temp_stress * 0.3)
                self.health_vector.apply_damage(HealthDimension.NEURAL, temp_stress * 0.2)
            else:  # Low temperature
                # Low temps primarily affect physical and immune systems
                self.health_vector.apply_damage(HealthDimension.PHYSICAL, temp_stress * 0.2)
                self.health_vector.apply_damage(HealthDimension.IMMUNE, temp_stress * 0.3)
        
        # Update behavior modifiers based on health state
        self._update_behavior_modifiers()
        
        # Return current health metrics
        return {
            "health": self.health_vector.to_dict(),
            "temperature": body_temp,
            "temperature_stress": temp_stress,
            "adapted_temperature": self.adapted_temperature,
            "allostatic_load": allostatic_load,
            "behavior_modifiers": self.behavior_modifiers
        }
    
    def _update_behavior_modifiers(self) -> None:
        """Update behavior modifiers based on current health state."""
        # Calculate overall health percentage
        overall_health = self.health_vector.get_overall_health() / 100.0
        
        # Update movement speed (affected by physical integrity)
        physical_factor = self.health_vector.physical_integrity / 100.0
        self.behavior_modifiers["movement_speed"] = 0.2 + physical_factor * 0.8
        
        # Update attack strength (affected by physical integrity and metabolic efficiency)
        attack_factor = (self.health_vector.physical_integrity * 0.7 + 
                        self.health_vector.metabolic_efficiency * 0.3) / 100.0
        self.behavior_modifiers["attack_strength"] = 0.3 + attack_factor * 0.7
        
        # Update gathering efficiency (affected by metabolic efficiency and physical integrity)
        gather_factor = (self.health_vector.metabolic_efficiency * 0.6 + 
                        self.health_vector.physical_integrity * 0.4) / 100.0
        self.behavior_modifiers["gather_efficiency"] = 0.3 + gather_factor * 0.7
        
        # Update decision quality (affected by neural health)
        decision_factor = self.health_vector.neural_health / 100.0
        self.behavior_modifiers["decision_quality"] = 0.5 + decision_factor * 0.5
    
    def take_damage(self, amount: float, target_dimension: Optional[HealthDimension] = None) -> float:
        """Apply damage to health vector.
        
        Args:
            amount: Amount of damage to apply
            target_dimension: Optional specific dimension to target (if None, applied to PHYSICAL)
            
        Returns:
            float: Actual amount of damage applied
        """
        # Default to physical damage if no dimension specified
        if target_dimension is None:
            target_dimension = HealthDimension.PHYSICAL
        
        # Apply damage to specified dimension
        return self.health_vector.apply_damage(target_dimension, amount)
    
    def is_alive(self) -> bool:
        """Check if agent is alive based on health state.
        
        Returns:
            bool: True if agent is alive, False if health conditions indicate death
        """
        # Check if any critical health dimension has reached zero
        if self.health_vector.physical_integrity <= 0:
            return False
            
        # Check if overall health is too low
        if self.health_vector.get_overall_health() < 10:
            return False
            
        return True
    
    def to_dict(self) -> Dict:
        """Convert health system state to dictionary for storage/serialization.
        
        Returns:
            Dict: Health system state as dictionary
        """
        return {
            "health_vector": self.health_vector.to_dict(),
            "homeostasis": {
                param.value: value for param, value in self.homeostasis.current_values.items()
            },
            "allostatic_load": self.allostatic_system.get_allostatic_load(),
            "adapted_temperature": self.adapted_temperature,
            "age": self.age,
            "non_repairable_damage": self.non_repairable_damage,
            "cellular_divisions": self.cellular_divisions,
            "behavior_modifiers": self.behavior_modifiers
        } 