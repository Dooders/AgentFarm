#!/usr/bin/env python
"""
Script to verify that the seed parameter in BaseDQNConfig produces reproducible results.
This script creates two DQN modules with the same seed and verifies that they:
1. Select the same actions given the same states
2. Produce the same training results
3. Have identical network weights after initialization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule

# Mock logger class to handle the log_learning_experience call
class MockLogger:
    def log_learning_experience(self, **kwargs):
        pass
    
    def batch_log_learning_experiences(self, experiences):
        pass

# Mock database class to provide the logger
class MockDatabase:
    def __init__(self):
        self.logger = MockLogger()

def verify_seed_reproducibility(seed=42):
    """Verify that using the same seed produces reproducible results."""
    print(f"Verifying seed reproducibility with seed={seed}")
    
    # Set numpy seed for consistent test state generation
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create two configs with the same seed
    config1 = BaseDQNConfig()
    config1.seed = seed
    config2 = BaseDQNConfig()
    config2.seed = seed
    
    # Simple environment parameters
    input_dim = 4
    output_dim = 2
    
    # Create mock database for logger
    mock_db = MockDatabase()
    
    # Create two DQN modules with the same seed
    dqn1 = BaseDQNModule(input_dim, output_dim, config1, db=mock_db)
    dqn2 = BaseDQNModule(input_dim, output_dim, config2, db=mock_db)
    
    # Verify network weights are identical after initialization
    print("\nVerifying network weights are identical after initialization...")
    weights_match = verify_network_weights(dqn1, dqn2)
    print(f"Network weights match: {weights_match}")
    
    # Verify action selection is identical
    print("\nVerifying action selection is identical...")
    actions_match = verify_action_selection(dqn1, dqn2, num_tests=10, seed=seed)
    print(f"Action selection matches: {actions_match}")
    
    # Verify training results are identical
    print("\nVerifying training results are identical...")
    training_matches = verify_training(dqn1, dqn2, num_steps=100, seed=seed)
    print(f"Training results match: {training_matches}")
    
    return weights_match and actions_match and training_matches

def verify_network_weights(dqn1, dqn2):
    """Verify that network weights are identical between two DQN modules."""
    for (name1, param1), (name2, param2) in zip(
        dqn1.q_network.named_parameters(), dqn2.q_network.named_parameters()
    ):
        if not torch.allclose(param1, param2):
            print(f"Weights don't match for {name1}")
            return False
    return True

def verify_action_selection(dqn1, dqn2, num_tests=10, seed=42):
    """Verify that action selection is identical between two DQN modules."""
    # Set seed for reproducible state generation
    torch.manual_seed(seed)
    
    for i in range(num_tests):
        # Create a random state (using the seeded random generator)
        state = torch.rand(dqn1.q_network.network[0].in_features)
        
        # Get actions from both networks
        action1 = dqn1.select_action(state, epsilon=0.0)  # Disable exploration
        action2 = dqn2.select_action(state, epsilon=0.0)  # Disable exploration
        
        if action1 != action2:
            print(f"Test {i}: Actions don't match! {action1} vs {action2}")
            return False
        else:
            print(f"Test {i}: Actions match: {action1}")
    
    return True

def verify_training(dqn1, dqn2, num_steps=100, seed=42):
    """Verify that training produces identical results between two DQN modules."""
    # Set seed for reproducible experience generation
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Ensure deterministic operations for training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Generate all experiences in advance to ensure identical batches
    experiences = []
    for step in range(num_steps):
        state = torch.rand(dqn1.q_network.network[0].in_features)
        action = 0 if np.random.rand() < 0.5 else 1
        reward = np.random.rand()
        next_state = torch.rand(dqn1.q_network.network[0].in_features)
        done = False
        
        experiences.append((state, action, reward, next_state, done, step))
    
    # Store identical experiences in both networks
    for state, action, reward, next_state, done, step in experiences:
        dqn1.store_experience(
            state, action, reward, next_state, done, 
            step_number=step, agent_id="test_agent", 
            module_type="test", module_id=1, 
            action_taken_mapped="test_action"
        )
        dqn2.store_experience(
            state, action, reward, next_state, done, 
            step_number=step, agent_id="test_agent", 
            module_type="test", module_id=1, 
            action_taken_mapped="test_action"
        )
    
    # Train both networks with identical batches
    for step in range(num_steps - dqn1.config.batch_size + 1):
        if len(dqn1.memory) >= dqn1.config.batch_size:
            # Create identical batches for both networks
            batch = list(dqn1.memory)[step:step+dqn1.config.batch_size]
            
            # Reset random state before each training step
            torch.manual_seed(seed + step)
            loss1 = dqn1.train(batch)
            
            torch.manual_seed(seed + step)
            loss2 = dqn2.train(batch)
            
            if loss1 is not None and loss2 is not None:
                if abs(loss1 - loss2) > 1e-5:
                    print(f"Step {step}: Losses don't match! {loss1} vs {loss2}")
                    print(f"Difference: {abs(loss1 - loss2)}")
                    return False
                else:
                    print(f"Step {step}: Losses match: {loss1}")
    
    # Compare final network weights
    return verify_network_weights(dqn1, dqn2)

def run_with_different_seeds():
    """Run experiments with different seeds to show they produce different results."""
    print("\nRunning with different seeds to verify they produce different results...")
    
    # Create mock database for logger
    mock_db = MockDatabase()
    
    # Create two configs with different seeds
    config1 = BaseDQNConfig()
    config1.seed = 42
    config2 = BaseDQNConfig()
    config2.seed = 43
    
    # Simple environment parameters
    input_dim = 4
    output_dim = 2
    
    # Create two DQN modules with different seeds
    dqn1 = BaseDQNModule(input_dim, output_dim, config1, db=mock_db)
    dqn2 = BaseDQNModule(input_dim, output_dim, config2, db=mock_db)
    
    # Verify network weights are different after initialization
    weights_match = verify_network_weights(dqn1, dqn2)
    print(f"Network weights match with different seeds: {weights_match}")
    print("(Should be False if seeds are working correctly)")

def plot_epsilon_decay():
    """Plot epsilon decay to visualize deterministic behavior."""
    print("\nPlotting epsilon decay to visualize deterministic behavior...")
    
    # Create mock database for logger
    mock_db = MockDatabase()
    
    # Create two configs with the same seed
    config1 = BaseDQNConfig()
    config1.seed = 42
    config1.epsilon_decay = 0.9  # Faster decay for visualization
    
    config2 = BaseDQNConfig()
    config2.seed = 42
    config2.epsilon_decay = 0.9  # Faster decay for visualization
    
    # Simple environment parameters
    input_dim = 4
    output_dim = 2
    
    # Create two DQN modules with the same seed
    dqn1 = BaseDQNModule(input_dim, output_dim, config1, db=mock_db)
    dqn2 = BaseDQNModule(input_dim, output_dim, config2, db=mock_db)
    
    # Track epsilon values over training steps
    epsilon_values1 = [dqn1.epsilon]
    epsilon_values2 = [dqn2.epsilon]
    
    # Set seed for reproducible experience generation
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate all experiences in advance
    experiences = []
    for step in range(50):
        state = torch.rand(input_dim)
        action = 0
        reward = 0.5
        next_state = torch.rand(input_dim)
        done = False
        experiences.append((state, action, reward, next_state, done, step))
    
    # Run some training steps
    for state, action, reward, next_state, done, step in experiences:
        # Store experiences
        dqn1.store_experience(
            state, action, reward, next_state, done,
            step_number=step, agent_id="test_agent", 
            module_type="test", module_id=1, 
            action_taken_mapped="test_action"
        )
        dqn2.store_experience(
            state, action, reward, next_state, done,
            step_number=step, agent_id="test_agent", 
            module_type="test", module_id=1, 
            action_taken_mapped="test_action"
        )
        
        # Train if enough experiences
        if len(dqn1.memory) >= dqn1.config.batch_size:
            # Reset random state before each training step
            torch.manual_seed(42 + step)
            batch1 = list(dqn1.memory)[-dqn1.config.batch_size:]
            dqn1.train(batch1)
            
            torch.manual_seed(42 + step)
            batch2 = list(dqn2.memory)[-dqn2.config.batch_size:]
            dqn2.train(batch2)
        
        # Track epsilon values
        epsilon_values1.append(dqn1.epsilon)
        epsilon_values2.append(dqn2.epsilon)
    
    # Plot epsilon decay
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values1, label='DQN 1 (seed=42)')
    plt.plot(epsilon_values2, label='DQN 2 (seed=42)', linestyle='--')
    plt.xlabel('Training Steps')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay with Same Seed')
    plt.legend()
    plt.grid(True)
    plt.savefig('epsilon_decay.png')
    plt.close()
    
    print("Epsilon decay plot saved as 'epsilon_decay.png'")
    
    # Verify the epsilon values are identical
    epsilon_match = all(abs(e1 - e2) < 1e-10 for e1, e2 in zip(epsilon_values1, epsilon_values2))
    print(f"Epsilon values match: {epsilon_match}")

if __name__ == "__main__":
    print("DQN Seed Verification Script")
    print("============================")
    
    # Verify seed reproducibility
    success = verify_seed_reproducibility(seed=42)
    
    # Run with different seeds to show they produce different results
    run_with_different_seeds()
    
    # Plot epsilon decay
    plot_epsilon_decay()
    
    if success:
        print("\n✅ SUCCESS: Seed parameter is working correctly!")
    else:
        print("\n❌ FAILURE: Seed parameter is not producing reproducible results.") 