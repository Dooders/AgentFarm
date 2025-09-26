/**
 * Performance validation tests for Zod Schema Validation System
 * Ensures schema validation meets performance requirements for acceptance criteria
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { validateSimulationConfig, getDefaultConfig } from '@/utils/validationUtils'
import { SimulationConfigSchema } from '@/types/zodSchemas'

describe('Schema Validation Performance Requirements', () => {
  let validConfig: ReturnType<typeof getDefaultConfig>
  let largeConfig: any

  beforeEach(() => {
    validConfig = getDefaultConfig()

    // Create a large configuration for stress testing
    largeConfig = {
      ...validConfig,
      width: 800,
      height: 600,
      system_agents: 5000,
      independent_agents: 5000,
      control_agents: 3000,
      agent_parameters: {
        SystemAgent: {
          ...validConfig.agent_parameters.SystemAgent,
          memory_size: 50000,
          dqn_hidden_size: 512,
          batch_size: 128
        },
        IndependentAgent: {
          ...validConfig.agent_parameters.IndependentAgent,
          memory_size: 50000,
          dqn_hidden_size: 512,
          batch_size: 128
        },
        ControlAgent: {
          ...validConfig.agent_parameters.ControlAgent,
          memory_size: 50000,
          dqn_hidden_size: 512,
          batch_size: 128
        }
      },
      gather_parameters: {
        ...validConfig.gather_parameters,
        memory_size: 25000,
        dqn_hidden_size: 256,
        batch_size: 64
      },
      share_parameters: {
        ...validConfig.share_parameters,
        memory_size: 25000,
        dqn_hidden_size: 256,
        batch_size: 64
      },
      move_parameters: {
        ...validConfig.move_parameters,
        memory_size: 25000,
        dqn_hidden_size: 256,
        batch_size: 64
      },
      attack_parameters: {
        ...validConfig.attack_parameters,
        memory_size: 25000,
        dqn_hidden_size: 256,
        batch_size: 64
      }
    }
  })

  describe('Single Configuration Validation Performance', () => {
    it('should validate a typical configuration in under 5ms', () => {
      const start = performance.now()
      const result = validateSimulationConfig(validConfig)
      const end = performance.now()

      const duration = end - start

      console.log(`Single config validation: ${duration.toFixed(3)}ms`)

      expect(result.success).toBe(true)
      expect(duration).toBeLessThan(5)
    })

    it('should validate a large configuration in under 10ms', () => {
      const start = performance.now()
      const result = validateSimulationConfig(largeConfig)
      const end = performance.now()

      const duration = end - start

      console.log(`Large config validation: ${duration.toFixed(3)}ms`)

      // Note: Large config should fail validation due to performance constraints,
      // but should still complete quickly
      expect(duration).toBeLessThan(10)
    })

    it('should validate configurations with consistent performance', () => {
      const iterations = 50
      const times: number[] = []

      for (let i = 0; i < iterations; i++) {
        const start = performance.now()
        validateSimulationConfig(validConfig)
        const end = performance.now()
        times.push(end - start)
      }

      const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length
      const maxTime = Math.max(...times)
      const minTime = Math.min(...times)
      const standardDeviation = Math.sqrt(
        times.reduce((sum, time) => sum + Math.pow(time - averageTime, 2), 0) / times.length
      )

      console.log(`Performance stats (${iterations} iterations):`)
      console.log(`  Average: ${averageTime.toFixed(3)}ms`)
      console.log(`  Min: ${minTime.toFixed(3)}ms`)
      console.log(`  Max: ${maxTime.toFixed(3)}ms`)
      console.log(`  Std Dev: ${standardDeviation.toFixed(3)}ms`)

      // Performance requirements
      expect(averageTime).toBeLessThan(5)
      expect(maxTime).toBeLessThan(10)
      expect(standardDeviation).toBeLessThan(2) // Consistent performance
    })
  })

  describe('Batch Validation Performance', () => {
    it('should validate 100 configurations in under 200ms', () => {
      const configs = Array.from({ length: 100 }, () => getDefaultConfig())
      const start = performance.now()

      for (const config of configs) {
        validateSimulationConfig(config)
      }

      const end = performance.now()
      const totalTime = end - start

      console.log(`100 config batch validation: ${totalTime.toFixed(2)}ms`)

      expect(totalTime).toBeLessThan(200) // 2ms average
    })

    it('should validate 1000 configurations in under 1 second', () => {
      const configs = Array.from({ length: 1000 }, () => getDefaultConfig())
      const start = performance.now()

      for (const config of configs) {
        validateSimulationConfig(config)
      }

      const end = performance.now()
      const totalTime = end - start

      console.log(`1000 config batch validation: ${totalTime.toFixed(2)}ms`)

      expect(totalTime).toBeLessThan(1000) // 1ms average
    })

    it('should handle mixed valid/invalid configurations efficiently', () => {
      const configs = []

      // Add 50 valid configs
      for (let i = 0; i < 50; i++) {
        configs.push(getDefaultConfig())
      }

      // Add 50 invalid configs
      for (let i = 0; i < 50; i++) {
        configs.push({
          ...getDefaultConfig(),
          width: i % 2 === 0 ? 5 : 1500, // Invalid values
          agent_type_ratios: { SystemAgent: 0.5, IndependentAgent: 0.3, ControlAgent: 0.1 } // Invalid sum
        })
      }

      const start = performance.now()

      for (const config of configs) {
        validateSimulationConfig(config)
      }

      const end = performance.now()
      const totalTime = end - start

      console.log(`Mixed validation batch (100 configs): ${totalTime.toFixed(2)}ms`)

      expect(totalTime).toBeLessThan(300) // 3ms average
    })
  })

  describe('Memory Efficiency', () => {
    it('should not cause memory leaks during repeated validation', () => {
      const initialMemory = performance.memory ? performance.memory.usedJSHeapSize : 0

      // Run many validations
      for (let i = 0; i < 1000; i++) {
        validateSimulationConfig(validConfig)
        validateSimulationConfig(largeConfig)
      }

      // Force garbage collection if available
      if (typeof global !== 'undefined' && global.gc) {
        global.gc()
      }

      const finalMemory = performance.memory ? performance.memory.usedJSHeapSize : 0
      const memoryIncrease = finalMemory - initialMemory

      console.log(`Memory usage - Initial: ${initialMemory}, Final: ${finalMemory}, Increase: ${memoryIncrease}`)

      // Allow for some memory increase but not excessive
      if (initialMemory > 0) {
        const memoryIncreasePercentage = (memoryIncrease / initialMemory) * 100
        expect(memoryIncreasePercentage).toBeLessThan(50) // Less than 50% memory increase
      }
    })

    it('should efficiently handle schema creation and reuse', () => {
      const start = performance.now()

      // Create multiple schemas (simulating component re-renders)
      for (let i = 0; i < 100; i++) {
        const schema = SimulationConfigSchema
        schema.parse(validConfig)
      }

      const end = performance.now()
      const duration = end - start

      console.log(`Schema reuse test (100 iterations): ${duration.toFixed(2)}ms`)

      expect(duration).toBeLessThan(100) // Very fast schema reuse
    })
  })

  describe('Scalability Tests', () => {
    it('should handle increasingly complex configurations', () => {
      const complexities = [
        { agents: 10, memory: 1000, hidden: 32 },
        { agents: 100, memory: 10000, hidden: 64 },
        { agents: 1000, memory: 50000, hidden: 128 },
        { agents: 5000, memory: 100000, hidden: 256 }
      ]

      const results = complexities.map(complexity => {
        const config = {
          ...validConfig,
          system_agents: complexity.agents,
          independent_agents: complexity.agents,
          control_agents: Math.floor(complexity.agents / 2),
          agent_parameters: {
            SystemAgent: {
              ...validConfig.agent_parameters.SystemAgent,
              memory_size: complexity.memory,
              dqn_hidden_size: complexity.hidden
            },
            IndependentAgent: {
              ...validConfig.agent_parameters.IndependentAgent,
              memory_size: complexity.memory,
              dqn_hidden_size: complexity.hidden
            },
            ControlAgent: {
              ...validConfig.agent_parameters.ControlAgent,
              memory_size: complexity.memory,
              dqn_hidden_size: complexity.hidden
            }
          }
        }

        const start = performance.now()
        const result = validateSimulationConfig(config)
        const end = performance.now()

        return {
          complexity: `agents: ${complexity.agents}, memory: ${complexity.memory}`,
          duration: end - start,
          valid: result.success
        }
      })

      results.forEach(result => {
        console.log(`Complexity ${result.complexity}: ${result.duration.toFixed(3)}ms, Valid: ${result.valid}`)
        expect(result.duration).toBeLessThan(15) // Allow slightly more time for complex configs
      })
    })

    it('should maintain performance under load', () => {
      const concurrentValidations = 10
      const promises: Promise<void>[] = []

      for (let i = 0; i < concurrentValidations; i++) {
        promises.push(
          new Promise((resolve) => {
            // Simulate async validation work
            setTimeout(() => {
              validateSimulationConfig(validConfig)
              resolve()
            }, Math.random() * 10) // Random delay to simulate real-world usage
          })
        )
      }

      const start = performance.now()

      return Promise.all(promises).then(() => {
        const end = performance.now()
        const totalTime = end - start

        console.log(`Concurrent validation test (${concurrentValidations} parallel): ${totalTime.toFixed(2)}ms`)

        // Should still be reasonably fast even with concurrent operations
        expect(totalTime).toBeLessThan(50)
      })
    })
  })

  describe('Performance Benchmarks', () => {
    it('should meet minimum performance requirements', () => {
      const benchmarks = {
        singleValidation: { target: 5, description: 'Single config validation' },
        hundredValidations: { target: 200, description: '100 config batch' },
        thousandValidations: { target: 1000, description: '1000 config batch' },
        largeConfigValidation: { target: 10, description: 'Large config validation' }
      }

      const results: Array<{ name: string; actual: number; target: number; passed: boolean }> = []

      // Single validation
      const singleStart = performance.now()
      validateSimulationConfig(validConfig)
      const singleTime = performance.now() - singleStart
      results.push({
        name: 'singleValidation',
        actual: singleTime,
        target: benchmarks.singleValidation.target,
        passed: singleTime < benchmarks.singleValidation.target
      })

      // Hundred validations
      const hundredStart = performance.now()
      const hundredConfigs = Array.from({ length: 100 }, () => getDefaultConfig())
      for (const config of hundredConfigs) {
        validateSimulationConfig(config)
      }
      const hundredTime = performance.now() - hundredStart
      results.push({
        name: 'hundredValidations',
        actual: hundredTime,
        target: benchmarks.hundredValidations.target,
        passed: hundredTime < benchmarks.hundredValidations.target
      })

      // Large config validation
      const largeStart = performance.now()
      validateSimulationConfig(largeConfig)
      const largeTime = performance.now() - largeStart
      results.push({
        name: 'largeConfigValidation',
        actual: largeTime,
        target: benchmarks.largeConfigValidation.target,
        passed: largeTime < benchmarks.largeConfigValidation.target
      })

      // Log results
      results.forEach(result => {
        console.log(`${result.name}: ${result.actual.toFixed(3)}ms (target: ${result.target}ms) ${result.passed ? '✅' : '❌'}`)
        expect(result.passed).toBe(true)
      })

      // Overall performance assessment
      const allPassed = results.every(r => r.passed)
      expect(allPassed).toBe(true)
    })
  })
})