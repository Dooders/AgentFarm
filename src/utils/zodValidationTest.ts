/**
 * Test file to validate Zod schema implementation
 * This file demonstrates the validation system and can be used for testing
 */

import { validateSimulationConfig, getDefaultConfig } from '@/utils/validationUtils'

// Test valid configuration
export function testValidConfig() {
  const validConfig = {
    width: 100,
    height: 100,
    position_discretization_method: 'floor' as const,
    use_bilinear_interpolation: true,
    system_agents: 10,
    independent_agents: 15,
    control_agents: 5,
    agent_type_ratios: {
      SystemAgent: 0.3,
      IndependentAgent: 0.5,
      ControlAgent: 0.2
    },
    learning_rate: 0.001,
    epsilon_start: 1.0,
    epsilon_min: 0.01,
    epsilon_decay: 0.995,
    agent_parameters: {
      SystemAgent: {
        target_update_freq: 100,
        memory_size: 10000,
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_min: 0.01,
        epsilon_decay: 0.995,
        dqn_hidden_size: 128,
        batch_size: 32,
        tau: 0.005,
        success_reward: 1.0,
        failure_penalty: -1.0,
        base_cost: 0.1
      },
      IndependentAgent: {
        target_update_freq: 100,
        memory_size: 10000,
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_min: 0.01,
        epsilon_decay: 0.995,
        dqn_hidden_size: 128,
        batch_size: 32,
        tau: 0.005,
        success_reward: 1.0,
        failure_penalty: -1.0,
        base_cost: 0.1
      },
      ControlAgent: {
        target_update_freq: 100,
        memory_size: 10000,
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_min: 0.01,
        epsilon_decay: 0.995,
        dqn_hidden_size: 128,
        batch_size: 32,
        tau: 0.005,
        success_reward: 1.0,
        failure_penalty: -1.0,
        base_cost: 0.1
      }
    },
    visualization: {
      canvas_width: 800,
      canvas_height: 600,
      background_color: '#1a1a1a',
      agent_colors: {
        SystemAgent: '#ff6b6b',
        IndependentAgent: '#4ecdc4',
        ControlAgent: '#45b7d1'
      },
      show_metrics: true,
      font_size: 12,
      line_width: 2
    },
    gather_parameters: {
      target_update_freq: 50,
      memory_size: 5000,
      learning_rate: 0.001,
      gamma: 0.99,
      epsilon_start: 1.0,
      epsilon_min: 0.01,
      epsilon_decay: 0.995,
      dqn_hidden_size: 64,
      batch_size: 16,
      tau: 0.005,
      success_reward: 0.5,
      failure_penalty: -0.5,
      base_cost: 0.05
    },
    share_parameters: {
      target_update_freq: 50,
      memory_size: 5000,
      learning_rate: 0.001,
      gamma: 0.99,
      epsilon_start: 1.0,
      epsilon_min: 0.01,
      epsilon_decay: 0.995,
      dqn_hidden_size: 64,
      batch_size: 16,
      tau: 0.005,
      success_reward: 0.5,
      failure_penalty: -0.5,
      base_cost: 0.05
    },
    move_parameters: {
      target_update_freq: 50,
      memory_size: 5000,
      learning_rate: 0.001,
      gamma: 0.99,
      epsilon_start: 1.0,
      epsilon_min: 0.01,
      epsilon_decay: 0.995,
      dqn_hidden_size: 64,
      batch_size: 16,
      tau: 0.005,
      success_reward: 0.5,
      failure_penalty: -0.5,
      base_cost: 0.05
    },
    attack_parameters: {
      target_update_freq: 50,
      memory_size: 5000,
      learning_rate: 0.001,
      gamma: 0.99,
      epsilon_start: 1.0,
      epsilon_min: 0.01,
      epsilon_decay: 0.995,
      dqn_hidden_size: 64,
      batch_size: 16,
      tau: 0.005,
      success_reward: 0.5,
      failure_penalty: -0.5,
      base_cost: 0.05
    }
  }

  const result = validateSimulationConfig(validConfig)

  if (result.success) {
    console.log('✅ Valid configuration passed validation')
    return true
  } else {
    console.log('❌ Valid configuration failed validation:', result.errors)
    return false
  }
}

// Test invalid configurations
export function testInvalidConfigs() {
  const invalidConfigs = [
    // Invalid width (too small)
    {
      name: 'Width too small',
      config: { width: 5, height: 100 },
      expectedErrors: ['width must be at least 10']
    },
    // Invalid agent ratios (don't sum to 1.0)
    {
      name: 'Invalid agent ratios',
      config: {
        agent_type_ratios: { SystemAgent: 0.5, IndependentAgent: 0.5, ControlAgent: 0.5 }
      },
      expectedErrors: ['Agent type ratios must sum to exactly 1.0']
    },
    // Invalid epsilon values
    {
      name: 'Invalid epsilon values',
      config: { epsilon_min: 0.5, epsilon_start: 0.3 },
      expectedErrors: ['Epsilon minimum must be less than or equal to epsilon start']
    },
    // Invalid hex color
    {
      name: 'Invalid hex color',
      config: {
        visualization: {
          background_color: 'invalid-color'
        }
      },
      expectedErrors: ['Background color must be a valid hex color code']
    }
  ]

  let passedTests = 0

  for (const testCase of invalidConfigs) {
    const baseConfig = getDefaultConfig()
    const testConfig = { ...baseConfig, ...testCase.config }

    const result = validateSimulationConfig(testConfig)

    if (!result.success && result.errors.length > 0) {
      console.log(`✅ ${testCase.name} correctly failed validation`)
      passedTests++
    } else {
      console.log(`❌ ${testCase.name} should have failed but passed`)
    }
  }

  console.log(`Validation tests: ${passedTests}/${invalidConfigs.length} passed`)
  return passedTests === invalidConfigs.length
}

// Run all validation tests
export function runValidationTests() {
  console.log('Running Zod Schema Validation Tests...\n')

  const validTestPassed = testValidConfig()
  const invalidTestsPassed = testInvalidConfigs()

  if (validTestPassed && invalidTestsPassed) {
    console.log('\n🎉 All validation tests passed!')
    return true
  } else {
    console.log('\n❌ Some validation tests failed')
    return false
  }
}

// Export test utilities for use in other test files
export const testUtils = {
  testValidConfig,
  testInvalidConfigs,
  runValidationTests
}