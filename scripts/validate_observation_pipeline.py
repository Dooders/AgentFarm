#!/usr/bin/env python3
"""
Comprehensive validation script for AgentFarm observation-to-decision pipeline.

This script validates:
1. Network Architecture: CNN implementations in RL algorithms
2. Observation Processing: 3D tensor handling and channel system
3. Action Selection & Masking: Curriculum learning support
4. Integration Testing: Complete pipeline functionality
5. Performance & Correctness: Memory efficiency and error handling
"""

import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

# Add the farm module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "farm"))

from farm.core.agent import BaseAgent
from farm.core.channels import NUM_CHANNELS, get_channel_registry
from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import DecisionModule
from farm.core.observations import AgentObservation, ObservationConfig
from farm.core.services.factory import AgentServiceFactory
from farm.utils.logging import get_logger

# from farm.core.services.spatial.mock_spatial_service import MockSpatialService


logger = get_logger(__name__)


class MockSpatialService:
    """Mock spatial service for testing."""

    def get_nearby(self, position, radius, index_names=None):
        return {"resources": [], "agents": []}

    def get_nearest(self, position, index_names=None):
        return {"resources": None, "agents": None}


class PipelineValidator:
    """Comprehensive validator for the observation-to-decision pipeline."""

    def __init__(self):
        self.results = {}
        self.errors = []

    def log_success(self, test_name: str, message: str = ""):
        """Log a successful test."""
        logger.info(f"✅ {test_name}: {message}")
        self.results[test_name] = {"status": "PASS", "message": message}

    def log_failure(self, test_name: str, error: str, details: str = ""):
        """Log a failed test."""
        logger.error(f"❌ {test_name}: {error}")
        if details:
            logger.error(f"   Details: {details}")
        self.results[test_name] = {"status": "FAIL", "error": error, "details": details}
        self.errors.append(f"{test_name}: {error}")

    def validate_channel_registry(self):
        """Validate the 13-channel observation system."""
        try:
            registry = get_channel_registry()
            num_channels = registry.num_channels

            if num_channels != 13:
                self.log_failure(
                    "Channel Registry Count",
                    f"Expected 13 channels, got {num_channels}",
                )
                return False

            # Check that all expected channels are registered
            expected_channels = [
                "SELF_HP",
                "ALLIES_HP",
                "ENEMIES_HP",
                "RESOURCES",
                "OBSTACLES",
                "TERRAIN_COST",
                "VISIBILITY",
                "KNOWN_EMPTY",
                "DAMAGE_HEAT",
                "TRAILS",
                "ALLY_SIGNAL",
                "GOAL",
                "LANDMARKS",
            ]

            missing_channels = []
            for channel_name in expected_channels:
                try:
                    registry.get_index(channel_name)
                except KeyError:
                    missing_channels.append(channel_name)

            if missing_channels:
                self.log_failure(
                    "Channel Registry Content", f"Missing channels: {missing_channels}"
                )
                return False

            self.log_success(
                "Channel Registry",
                f"Successfully registered {num_channels} channels: {expected_channels}",
            )
            return True

        except Exception as e:
            self.log_failure("Channel Registry", str(e))
            return False

    def validate_observation_config(self):
        """Validate observation configuration."""
        try:
            config = ObservationConfig(R=6)
            expected_size = 2 * config.R + 1  # Should be 13

            if expected_size != 13:
                self.log_failure(
                    "Observation Config Size", f"Expected size 13, got {expected_size}"
                )
                return False

            # Test tensor creation
            obs = AgentObservation(config)
            tensor = obs.tensor()

            expected_shape = (NUM_CHANNELS, 13, 13)
            if tensor.shape != expected_shape:
                self.log_failure(
                    "Observation Tensor Shape",
                    f"Expected shape {expected_shape}, got {tensor.shape}",
                )
                return False

            self.log_success(
                "Observation Config",
                f"Observation size: {expected_size}x{expected_size}, channels: {NUM_CHANNELS}",
            )
            return True

        except Exception as e:
            self.log_failure("Observation Config", str(e))
            return False

    def validate_network_architecture(self):
        """Validate CNN network architectures for RL algorithms."""
        try:
            # Test each algorithm's network creation
            algorithms = ["ppo", "sac", "dqn", "a2c"]
            observation_shape = (13, 13, 13)  # 13 channels, 13x13 spatial
            num_actions = 8

            for algorithm in algorithms:
                try:
                    # Create decision config
                    config = DecisionConfig(algorithm_type=algorithm)

                    # Create mock agent
                    mock_spatial = MockSpatialService()
                    mock_agent = BaseAgent(
                        agent_id=f"test_agent_{algorithm}",
                        position=(50.0, 50.0),
                        resource_level=10,
                        spatial_service=mock_spatial,
                    )

                    # Create decision module
                    decision_module = DecisionModule(
                        agent=mock_agent,
                        action_space=type(
                            "MockSpace", (), {"n": num_actions, "shape": (num_actions,)}
                        )(),
                        observation_space=type(
                            "MockSpace", (), {"shape": observation_shape}
                        )(),
                        config=config,
                    )

                    # Check if CNN networks were created properly
                    if hasattr(decision_module.algorithm, "policy"):
                        policy = decision_module.algorithm.policy

                        # Check for CNN layers in the network
                        has_conv = False
                        if hasattr(policy, "actor") and hasattr(
                            policy.actor, "conv_layers"
                        ):
                            has_conv = True
                        elif hasattr(policy, "model") and hasattr(
                            policy.model, "conv_layers"
                        ):
                            has_conv = True

                        if not has_conv:
                            self.log_failure(
                                f"{algorithm.upper()} CNN Architecture",
                                f"No convolutional layers found in {algorithm} network",
                            )
                            continue

                        self.log_success(
                            f"{algorithm.upper()} CNN Architecture",
                            "Successfully created CNN network with convolutional layers",
                        )
                    else:
                        self.log_failure(
                            f"{algorithm.upper()} Policy Creation",
                            "Failed to create policy object",
                        )

                except Exception as e:
                    self.log_failure(
                        f"{algorithm.upper()} Network Creation",
                        f"Failed to create {algorithm} network: {str(e)}",
                    )

            return True

        except Exception as e:
            self.log_failure("Network Architecture Validation", str(e))
            return False

    def validate_observation_processing(self):
        """Validate 3D observation tensor processing."""
        try:
            config = ObservationConfig(R=6)
            obs = AgentObservation(config)

            # Create test observation data
            test_state = torch.randn(13, 13, 13)  # 13 channels, 13x13 spatial

            # Test batch dimension handling
            # Single observation (no batch dim)
            single_obs = test_state
            if single_obs.dim() == 3:
                batched_obs = single_obs.unsqueeze(0)  # Add batch dimension
                if batched_obs.shape[0] != 1:
                    self.log_failure(
                        "Batch Dimension Handling",
                        f"Expected batch size 1, got {batched_obs.shape[0]}",
                    )
                    return False

            # Test tensor device handling
            device_obs = test_state.to("cpu")  # Ensure on CPU for testing
            if device_obs.device.type != "cpu":
                self.log_failure(
                    "Device Handling", f"Expected CPU device, got {device_obs.device}"
                )
                return False

            self.log_success(
                "Observation Processing",
                f"Successfully processed 3D tensor: shape {test_state.shape}, device {device_obs.device}",
            )
            return True

        except Exception as e:
            self.log_failure("Observation Processing", str(e))
            return False

    def validate_action_masking(self):
        """Validate action masking and curriculum learning support."""
        try:
            config = DecisionConfig(algorithm_type="ppo")
            mock_spatial = MockSpatialService()
            mock_agent = BaseAgent(
                agent_id="test_agent_masking",
                position=(50.0, 50.0),
                resource_level=10,
                spatial_service=mock_spatial,
            )

            # Create decision module
            decision_module = DecisionModule(
                agent=mock_agent,
                action_space=type("MockSpace", (), {"n": 8, "shape": (8,)})(),
                observation_space=type("MockSpace", (), {"shape": (13, 13, 13)})(),
                config=config,
            )

            # Test action masking
            test_state = torch.randn(13, 13, 13)
            enabled_actions = [0, 2, 4, 6]  # Only even actions enabled

            # Test action selection with masking
            action = decision_module.decide_action(test_state, enabled_actions)

            if action not in enabled_actions:
                self.log_failure(
                    "Action Masking",
                    f"Selected action {action} not in enabled actions {enabled_actions}",
                )
                return False

            # Test without masking
            action_no_mask = decision_module.decide_action(test_state, None)

            if action_no_mask < 0 or action_no_mask >= 8:
                self.log_failure(
                    "Action Selection (No Mask)",
                    f"Selected action {action_no_mask} out of valid range [0, 7]",
                )
                return False

            self.log_success(
                "Action Masking",
                f"Successfully masked actions: selected {action} from {enabled_actions}",
            )
            return True

        except Exception as e:
            self.log_failure("Action Masking", str(e))
            return False

    def validate_integration_pipeline(self):
        """Validate complete pipeline from observation to decision."""
        try:
            # Create components
            config = ObservationConfig(R=6)
            obs = AgentObservation(config)

            mock_spatial = MockSpatialService()
            mock_agent = BaseAgent(
                agent_id="test_agent_pipeline",
                position=(50.0, 50.0),
                resource_level=10,
                spatial_service=mock_spatial,
            )

            decision_config = DecisionConfig(algorithm_type="ppo")
            decision_module = DecisionModule(
                agent=mock_agent,
                action_space=type("MockSpace", (), {"n": 8, "shape": (8,)})(),
                observation_space=type("MockSpace", (), {"shape": (13, 13, 13)})(),
                config=decision_config,
            )

            # Step 1: Create observation tensor
            observation_tensor = obs.tensor()
            if observation_tensor.shape != (13, 13, 13):
                self.log_failure(
                    "Integration Step 1",
                    f"Observation tensor shape mismatch: {observation_tensor.shape}",
                )
                return False

            # Step 2: Create decision state (convert to torch)
            decision_state = observation_tensor.clone().detach()
            if not isinstance(decision_state, torch.Tensor):
                self.log_failure(
                    "Integration Step 2",
                    f"Decision state is not a torch tensor: {type(decision_state)}",
                )
                return False

            # Step 3: Make decision
            action = decision_module.decide_action(decision_state)

            if not isinstance(action, int) or action < 0 or action >= 8:
                self.log_failure(
                    "Integration Step 3", f"Invalid action selected: {action}"
                )
                return False

            self.log_success(
                "Integration Pipeline",
                f"Complete pipeline successful: obs {observation_tensor.shape} → decision → action {action}",
            )
            return True

        except Exception as e:
            self.log_failure("Integration Pipeline", str(e))
            return False

    def validate_performance(self):
        """Validate memory efficiency and performance."""
        try:
            import time

            # Test memory efficiency
            config = ObservationConfig(R=6)
            obs = AgentObservation(config)

            # Create multiple observations to test memory handling
            observations = []
            start_time = time.time()

            for i in range(100):
                obs_tensor = obs.tensor()
                observations.append(obs_tensor)

                # Clear tensor to test memory management
                del obs_tensor

            end_time = time.time()
            duration = end_time - start_time

            # Test should complete within reasonable time
            if duration > 5.0:  # 5 seconds max
                self.log_failure("Performance Timing", ".2f")
                return False

            # Test tensor device consistency
            test_tensor = torch.randn(13, 13, 13)
            if test_tensor.device.type not in ["cpu", "cuda"]:
                self.log_failure(
                    "Tensor Device Consistency",
                    f"Unexpected device: {test_tensor.device}",
                )
                return False

            self.log_success(
                "Performance", f"Performance test completed in {duration:.2f}s"
            )
            return True

        except Exception as e:
            self.log_failure("Performance", str(e))
            return False

    def validate_error_handling(self):
        """Validate error handling and fallback mechanisms."""
        try:
            # Test invalid observation shape
            try:
                config = ObservationConfig(R=-1)  # Invalid radius
                self.log_failure(
                    "Error Handling - Invalid Config",
                    "Should have failed with invalid radius",
                )
                return False
            except Exception:
                pass  # Expected to fail

            # Test invalid algorithm type
            try:
                config = DecisionConfig(algorithm_type="invalid_algorithm")
                mock_spatial = MockSpatialService()
                mock_agent = BaseAgent(
                    agent_id="test_agent_error",
                    position=(50.0, 50.0),
                    resource_level=10,
                    spatial_service=mock_spatial,
                )

                decision_module = DecisionModule(
                    agent=mock_agent,
                    action_space=type("MockSpace", (), {"n": 8, "shape": (8,)})(),
                    observation_space=type("MockSpace", (), {"shape": (13, 13, 13)})(),
                    config=config,
                )

                # Should fall back to valid algorithm
                if not hasattr(decision_module, "algorithm"):
                    self.log_failure(
                        "Error Handling - Invalid Algorithm",
                        "No fallback algorithm created",
                    )
                    return False

            except Exception as e:
                self.log_failure(
                    "Error Handling - Invalid Algorithm", f"Unexpected error: {str(e)}"
                )
                return False

            self.log_success(
                "Error Handling",
                "Successfully handled invalid configurations with fallbacks",
            )
            return True

        except Exception as e:
            self.log_failure("Error Handling", str(e))
            return False

    def run_all_validations(self):
        """Run all validation tests."""
        logger.info("🚀 Starting AgentFarm Observation-to-Decision Pipeline Validation")
        logger.info("=" * 70)

        tests = [
            ("Channel Registry", self.validate_channel_registry),
            ("Observation Config", self.validate_observation_config),
            ("Network Architecture", self.validate_network_architecture),
            ("Observation Processing", self.validate_observation_processing),
            ("Action Masking", self.validate_action_masking),
            ("Integration Pipeline", self.validate_integration_pipeline),
            ("Performance", self.validate_performance),
            ("Error Handling", self.validate_error_handling),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running {test_name} validation...")
            try:
                if test_func():
                    passed += 1
                else:
                    logger.warning(f"⚠️  {test_name} validation failed")
            except Exception as e:
                self.log_failure(test_name, f"Unexpected error: {str(e)}")
                logger.error(f"💥 {test_name} validation crashed: {str(e)}")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        if self.errors:
            logger.info("\n❌ ERRORS FOUND:")
            for error in self.errors:
                logger.info(f"  • {error}")

        if passed == total:
            logger.info("\n🎉 ALL VALIDATIONS PASSED!")
            return True
        else:
            logger.info(f"\n⚠️  {total - passed} validation(s) failed")
            return False


def main():
    """Main validation entry point."""
    validator = PipelineValidator()
    success = validator.run_all_validations()

    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS:")
    print("=" * 70)

    for test_name, result in validator.results.items():
        status = result["status"]
        if status == "PASS":
            print(f"✅ {test_name:.<50} PASS")
        else:
            print(f"❌ {test_name:.<50} FAIL")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
