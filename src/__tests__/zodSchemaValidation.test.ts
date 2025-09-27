/**
 * Comprehensive unit tests for Zod Schema Validation System
 * Validates all acceptance criteria for Issue #3
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import {
  SimulationConfigSchema,
  AgentParameterSchema,
  ModuleParameterSchema,
  VisualizationConfigSchema,
  AgentTypeRatiosSchema
} from '@/types/zodSchemas'
import {
  validateSimulationConfig,
  validateField,
  safeValidateConfig,
  getDefaultConfig,
  formatZodError
} from '@/utils/validationUtils'
import { z } from 'zod'

// Test data fixtures
const createValidConfig = () => ({
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
})

describe('Zod Schema Validation System - Acceptance Criteria Tests', () => {
  let validConfig: ReturnType<typeof createValidConfig>
  let startTime: number

  beforeEach(() => {
    validConfig = createValidConfig()
    startTime = Date.now()
  })

  afterEach(() => {
    const endTime = Date.now()
    const duration = endTime - startTime
    // Ensure validation performance is acceptable (< 50ms for entire test including setup)
    expect(duration).toBeLessThan(50)
  })

  describe('✅ All simulation config fields have Zod validation', () => {
    it('should validate all root-level configuration fields', async () => {
      const result = await validateSimulationConfig(validConfig)

      expect(result.success).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should reject invalid root-level field values', async () => {
      const invalidConfig = {
        ...validConfig,
        width: 5, // Too small
        height: 1500, // Too large
        system_agents: -1, // Negative
        learning_rate: 2.0 // Too large
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.length).toBeGreaterThan(0)
      expect(result.errors.some(e => e.path === 'width')).toBe(true)
      expect(result.errors.some(e => e.path === 'height')).toBe(true)
      expect(result.errors.some(e => e.path === 'system_agents')).toBe(true)
      expect(result.errors.some(e => e.path === 'learning_rate')).toBe(true)
    })

    it('should validate enum fields correctly', async () => {
      const invalidConfig = {
        ...validConfig,
        position_discretization_method: 'invalid' as any
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.message.includes('floor, round, or ceil')
      )).toBe(true)
    })
  })

  describe('✅ Nested object validation works correctly', () => {
    it('should validate nested agent parameters', async () => {
      const result = await validateSimulationConfig(validConfig)

      expect(result.success).toBe(true)

      // Test invalid nested parameters
      const invalidConfig = {
        ...validConfig,
        agent_parameters: {
          ...validConfig.agent_parameters,
          SystemAgent: {
            ...validConfig.agent_parameters.SystemAgent,
            memory_size: 500 // Too small
          }
        }
      }

      const nestedResult = await validateSimulationConfig(invalidConfig)

      expect(nestedResult.success).toBe(false)
      expect(nestedResult.errors.some(e =>
        e.path.includes('agent_parameters.SystemAgent.memory_size')
      )).toBe(true)
    })

    it('should validate nested visualization config', async () => {
      const invalidConfig = {
        ...validConfig,
        visualization: {
          ...validConfig.visualization,
          canvas_width: 200, // Too small
          background_color: 'invalid-color' // Invalid format
        }
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.path.includes('visualization.canvas_width')
      )).toBe(true)
      expect(result.errors.some(e =>
        e.path.includes('visualization.background_color')
      )).toBe(true)
    })

    it('should validate nested module parameters', async () => {
      const invalidConfig = {
        ...validConfig,
        gather_parameters: {
          ...validConfig.gather_parameters,
          batch_size: 2048 // Too large for module
        }
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.path.includes('gather_parameters.batch_size')
      )).toBe(true)
    })
  })

  describe('✅ Custom validation rules are implemented', () => {
    it('should enforce agent type ratios sum to 1.0', async () => {
      const invalidConfig = {
        ...validConfig,
        agent_type_ratios: {
          SystemAgent: 0.5,
          IndependentAgent: 0.5,
          ControlAgent: 0.5 // Sum = 1.5, should fail
        }
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.message.includes('sum to exactly 1.0')
      )).toBe(true)
    })

    it('should enforce epsilon hierarchy', async () => {
      const invalidConfig = {
        ...validConfig,
        epsilon_min: 0.5,
        epsilon_start: 0.3 // epsilon_min > epsilon_start, should fail
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.message.includes('Epsilon minimum must be less than or equal to epsilon start')
      )).toBe(true)
    })

    it('should validate performance constraints', async () => {
      const largeConfig = {
        ...validConfig,
        width: 1000,
        height: 1000,
        system_agents: 10000,
        independent_agents: 10000,
        control_agents: 10000
      }

      const result = await validateSimulationConfig(largeConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.message.includes('performance issues') ||
        e.message.includes('exceeds environment capacity')
      )).toBe(true)
    })
  })

  describe('✅ Error messages are properly formatted', () => {
    it('should provide user-friendly error messages', async () => {
      const invalidConfig = {
        ...validConfig,
        width: 'invalid' as any // String instead of number
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.message.includes('Expected number') ||
        e.message.includes('Expected a number') ||
        e.message.includes('Invalid type') ||
        e.message.includes('invalid_type')
      )).toBe(true)
    })

    it('should provide specific range error messages', async () => {
      const invalidConfig = {
        ...validConfig,
        width: 5 // Too small
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.message.includes('at least 10')
      )).toBe(true)
    })

    it('should provide format-specific error messages', async () => {
      const invalidConfig = {
        ...validConfig,
        visualization: {
          ...validConfig.visualization,
          background_color: 'not-a-hex-color'
        }
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.some(e =>
        e.message.includes('hex color code')
      )).toBe(true)
    })

    it('should include error codes for programmatic handling', async () => {
      const invalidConfig = {
        ...validConfig,
        width: 5
      }

      const result = await validateSimulationConfig(invalidConfig)

      expect(result.success).toBe(false)
      expect(result.errors.every(e => e.code && e.code.length > 0)).toBe(true)
    })
  })

  describe('✅ Schema validation performance is acceptable', () => {
    it('should validate configuration in under 5ms', async () => {
      const iterations = 100
      const times: number[] = []

      for (let i = 0; i < iterations; i++) {
        const iterationStart = performance.now()
        await validateSimulationConfig(validConfig)
        const iterationEnd = performance.now()
        times.push(iterationEnd - iterationStart)
      }

      const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length
      const maxTime = Math.max(...times)

      console.log(`Average validation time: ${averageTime.toFixed(3)}ms`)
      console.log(`Max validation time: ${maxTime.toFixed(3)}ms`)

      expect(averageTime).toBeLessThan(5)
      expect(maxTime).toBeLessThan(10)
    })

    it('should handle large configurations efficiently', async () => {
      const largeConfig = {
        ...validConfig,
        width: 500,
        height: 500,
        system_agents: 1000,
        independent_agents: 1000,
        control_agents: 1000,
        agent_parameters: {
          ...validConfig.agent_parameters,
          SystemAgent: { ...validConfig.agent_parameters.SystemAgent, memory_size: 100000 },
          IndependentAgent: { ...validConfig.agent_parameters.IndependentAgent, memory_size: 100000 },
          ControlAgent: { ...validConfig.agent_parameters.ControlAgent, memory_size: 100000 }
        }
      }

      const start = performance.now()
      const result = await validateSimulationConfig(largeConfig)
      const end = performance.now()

      expect(end - start).toBeLessThan(10)
      expect(result.success).toBe(false) // Should fail due to performance constraints
    })
  })

  describe('✅ TypeScript integration works seamlessly', () => {
    it('should provide full type safety with Zod schemas', async () => {
      // TypeScript should catch these at compile time, but we test at runtime
      const config: Parameters<typeof SimulationConfigSchema.parse>[0] = validConfig

      expect(() => {
        SimulationConfigSchema.parse(config)
      }).not.toThrow()

      // Test type inference
      const parsed = SimulationConfigSchema.parse(validConfig)
      expect(typeof parsed.width).toBe('number')
      expect(typeof parsed.agent_parameters.SystemAgent.learning_rate).toBe('number')
      expect(typeof parsed.visualization.background_color).toBe('string')
    })

    it('should work with safeValidateConfig returning typed data', async () => {
      const result = safeValidateConfig(validConfig)

      expect(result.success).toBe(true)
      if (result.success && result.data) {
        // TypeScript should know the exact type here
        const typedConfig = result.data
        expect(typedConfig.width).toBe(100)
        expect(typedConfig.agent_parameters.SystemAgent.memory_size).toBe(10000)
        expect(typedConfig.visualization.canvas_width).toBe(800)
      }
    })

    it('should provide proper IntelliSense support', async () => {
      // This test validates that all fields are properly typed
      const config = getDefaultConfig()

      // These should all be type-safe
      config.width = 200
      config.agent_parameters.SystemAgent.learning_rate = 0.01
      config.visualization.background_color = '#ffffff'
      // Adjust ratios to maintain sum of 1.0
      config.agent_type_ratios.SystemAgent = 0.4
      config.agent_type_ratios.IndependentAgent = 0.4
      config.agent_type_ratios.ControlAgent = 0.2

      const result = await validateSimulationConfig(config)
      expect(result.success).toBe(true)
    })
  })

  describe('Field-level validation', () => {
    it('should validate individual fields correctly', async () => {
      const widthErrors = validateField('width', 50)
      expect(widthErrors).toHaveLength(0)

      const invalidWidthErrors = validateField('width', 5)
      expect(invalidWidthErrors.length).toBeGreaterThan(0)
      expect(invalidWidthErrors[0].path).toBe('width')
    })

    it('should validate nested field paths', () => {
      const validNestedErrors = validateField('agent_parameters.SystemAgent.learning_rate', 0.001)
      expect(validNestedErrors).toHaveLength(0)

      const invalidNestedErrors = validateField('agent_parameters.SystemAgent.learning_rate', 2.0)
      expect(invalidNestedErrors.length).toBeGreaterThan(0)
      expect(invalidNestedErrors[0].path).toBe('agent_parameters.SystemAgent.learning_rate')
    })
  })

  describe('Default configuration', () => {
    it('should provide valid default configuration', async () => {
      const defaultConfig = getDefaultConfig()
      const result = await validateSimulationConfig(defaultConfig)

      expect(result.success).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should have all required fields in default config', () => {
      const defaultConfig = getDefaultConfig()

      expect(defaultConfig).toHaveProperty('width')
      expect(defaultConfig).toHaveProperty('height')
      expect(defaultConfig).toHaveProperty('agent_parameters')
      expect(defaultConfig).toHaveProperty('visualization')
      expect(defaultConfig).toHaveProperty('gather_parameters')
      expect(defaultConfig.agent_parameters).toHaveProperty('SystemAgent')
      expect(defaultConfig.agent_parameters).toHaveProperty('IndependentAgent')
      expect(defaultConfig.agent_parameters).toHaveProperty('ControlAgent')
    })
  })
})

// Performance benchmark test
describe('Performance Benchmarks', () => {
  it('should validate 1000 configurations in under 1 second', async () => {
    const configs = Array.from({ length: 1000 }, () => createValidConfig())
    const start = performance.now()

    for (const config of configs) {
      await validateSimulationConfig(config)
    }

    const end = performance.now()
    const totalTime = end - start

    console.log(`1000 validations completed in ${totalTime.toFixed(2)}ms`)
    expect(totalTime).toBeLessThan(1000) // Should be under 1 second
  })
})