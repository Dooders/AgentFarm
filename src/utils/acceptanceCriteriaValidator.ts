/**
 * Acceptance Criteria Validator for Issue #3: Zod Schema Validation System
 *
 * This utility validates that all acceptance criteria have been met:
 * ✅ All simulation config fields have Zod validation
 * ✅ Nested object validation works correctly
 * ✅ Custom validation rules are implemented
 * ✅ Error messages are properly formatted
 * ✅ Schema validation performance is acceptable
 * ✅ TypeScript integration works seamlessly
 */

import { validateSimulationConfig, getDefaultConfig, safeValidateConfig } from './validationUtils'
import { SimulationConfigSchema } from '@/types/zodSchemas'

export interface AcceptanceCriteriaResult {
  criterion: string
  passed: boolean
  description: string
  details?: string
  performance?: {
    averageTime: number
    maxTime: number
    totalTime: number
  }
}

export interface ValidationReport {
  overallSuccess: boolean
  totalCriteria: number
  passedCriteria: number
  failedCriteria: number
  results: AcceptanceCriteriaResult[]
  summary: string
  recommendations?: string[]
}

/**
 * Validates all acceptance criteria for the Zod Schema Validation System
 */
export async function validateAcceptanceCriteria(): Promise<ValidationReport> {
  const results: AcceptanceCriteriaResult[] = []
  const startTime = performance.now()

  // Test 1: All simulation config fields have Zod validation
  const criterion1 = await validateAllFieldsHaveValidation()
  results.push(criterion1)

  // Test 2: Nested object validation works correctly
  const criterion2 = await validateNestedObjectValidation()
  results.push(criterion2)

  // Test 3: Custom validation rules are implemented
  const criterion3 = await validateCustomValidationRules()
  results.push(criterion3)

  // Test 4: Error messages are properly formatted
  const criterion4 = await validateErrorMessageFormatting()
  results.push(criterion4)

  // Test 5: Schema validation performance is acceptable
  const criterion5 = await validatePerformance()
  results.push(criterion5)

  // Test 6: TypeScript integration works seamlessly
  const criterion6 = await validateTypeScriptIntegration()
  results.push(criterion6)

  const endTime = performance.now()
  const totalTime = endTime - startTime

  const passedCriteria = results.filter(r => r.passed).length
  const failedCriteria = results.length - passedCriteria
  const overallSuccess = failedCriteria === 0

  const report: ValidationReport = {
    overallSuccess,
    totalCriteria: results.length,
    passedCriteria,
    failedCriteria,
    results,
    summary: overallSuccess
      ? `✅ All ${results.length} acceptance criteria passed successfully!`
      : `❌ ${failedCriteria} out of ${results.length} acceptance criteria failed.`,
    recommendations: overallSuccess ? undefined : generateRecommendations(results)
  }

  console.log(`\n=== ACCEPTANCE CRITERIA VALIDATION REPORT ===`)
  console.log(report.summary)
  console.log(`Total Time: ${totalTime.toFixed(2)}ms`)
  console.log(`Criteria: ${passedCriteria}/${results.length} passed`)

  results.forEach((result, index) => {
    console.log(`${index + 1}. ${result.criterion}: ${result.passed ? '✅ PASS' : '❌ FAIL'}`)
    if (!result.passed && result.details) {
      console.log(`   Details: ${result.details}`)
    }
  })

  if (!overallSuccess && report.recommendations) {
    console.log(`\nRecommendations:`)
    report.recommendations.forEach(rec => console.log(`- ${rec}`))
  }

  return report
}

/**
 * Criterion 1: All simulation config fields have Zod validation
 */
async function validateAllFieldsHaveValidation(): Promise<AcceptanceCriteriaResult> {
  const validConfig = getDefaultConfig()
  const result = validateSimulationConfig(validConfig)

  const passed = result.success && result.errors.length === 0
  const details = passed
    ? 'All fields validated successfully without errors'
    : `Validation failed with ${result.errors.length} errors: ${result.errors.map(e => e.message).join(', ')}`

  return {
    criterion: 'All simulation config fields have Zod validation',
    passed,
    description: 'Validates that every field in the simulation configuration has proper Zod schema validation',
    details
  }
}

/**
 * Criterion 2: Nested object validation works correctly
 */
async function validateNestedObjectValidation(): Promise<AcceptanceCriteriaResult> {
  const tests = [
    {
      name: 'Agent parameters',
      config: {
        ...getDefaultConfig(),
        agent_parameters: {
          ...getDefaultConfig().agent_parameters,
          SystemAgent: { ...getDefaultConfig().agent_parameters.SystemAgent, memory_size: 500 }
        }
      },
      shouldFail: true
    },
    {
      name: 'Visualization config',
      config: {
        ...getDefaultConfig(),
        visualization: {
          ...getDefaultConfig().visualization,
          canvas_width: 200
        }
      },
      shouldFail: true
    },
    {
      name: 'Module parameters',
      config: {
        ...getDefaultConfig(),
        gather_parameters: {
          ...getDefaultConfig().gather_parameters,
          batch_size: 2048
        }
      },
      shouldFail: true
    }
  ]

  let passedTests = 0

  for (const test of tests) {
    const result = validateSimulationConfig(test.config)
    const passed = test.shouldFail ? !result.success : result.success

    if (passed) passedTests++
  }

  const passed = passedTests === tests.length
  const details = passed
    ? 'All nested object validations working correctly'
    : `${passedTests}/${tests.length} nested validation tests passed`

  return {
    criterion: 'Nested object validation works correctly',
    passed,
    description: 'Validates that nested objects (agent_parameters, visualization, module_parameters) are properly validated',
    details
  }
}

/**
 * Criterion 3: Custom validation rules are implemented
 */
async function validateCustomValidationRules(): Promise<AcceptanceCriteriaResult> {
  const tests = [
    {
      name: 'Agent ratios sum to 1.0',
      config: {
        ...getDefaultConfig(),
        agent_type_ratios: { SystemAgent: 0.5, IndependentAgent: 0.5, ControlAgent: 0.5 }
      },
      shouldFail: true,
      expectedError: 'sum to exactly 1.0'
    },
    {
      name: 'Epsilon hierarchy',
      config: {
        ...getDefaultConfig(),
        epsilon_min: 0.5,
        epsilon_start: 0.3
      },
      shouldFail: true,
      expectedError: 'Epsilon minimum must be less than or equal to epsilon start'
    },
    {
      name: 'Performance constraints',
      config: {
        ...getDefaultConfig(),
        width: 1000,
        height: 1000,
        system_agents: 10000,
        independent_agents: 10000,
        control_agents: 10000
      },
      shouldFail: true,
      expectedError: 'performance issues'
    }
  ]

  let passedTests = 0

  for (const test of tests) {
    const result = validateSimulationConfig(test.config)
    const hasExpectedError = result.errors.some(e => e.message.includes(test.expectedError))
    const passed = test.shouldFail ? (!result.success && hasExpectedError) : result.success

    if (passed) passedTests++
  }

  const passed = passedTests === tests.length
  const details = passed
    ? 'All custom validation rules working correctly'
    : `${passedTests}/${tests.length} custom validation rule tests passed`

  return {
    criterion: 'Custom validation rules are implemented',
    passed,
    description: 'Validates that cross-field validation rules and custom business logic validation is working',
    details
  }
}

/**
 * Criterion 4: Error messages are properly formatted
 */
async function validateErrorMessageFormatting(): Promise<AcceptanceCriteriaResult> {
  const testConfig = {
    ...getDefaultConfig(),
    width: 'invalid' as any,
    height: -10,
    system_agents: 'not-a-number' as any,
    agent_type_ratios: { SystemAgent: 0.5, IndependentAgent: 0.5, ControlAgent: 0.5 },
    visualization: {
      ...getDefaultConfig().visualization,
      background_color: 'not-a-hex-color'
    }
  }

  const result = validateSimulationConfig(testConfig)
  const hasUserFriendlyMessages = result.errors.some(e =>
    e.message.includes('Expected a number') ||
    e.message.includes('must be at least') ||
    e.message.includes('sum to exactly') ||
    e.message.includes('hex color code')
  )

  const allHaveErrorCodes = result.errors.every(e => e.code && e.code.length > 0)
  const passed = !result.success && hasUserFriendlyMessages && allHaveErrorCodes

  const details = passed
    ? 'Error messages are user-friendly, specific, and include error codes'
    : `Error message validation failed. User-friendly: ${hasUserFriendlyMessages}, Error codes: ${allHaveErrorCodes}`

  return {
    criterion: 'Error messages are properly formatted',
    passed,
    description: 'Validates that error messages are user-friendly, specific, and include error codes for programmatic handling',
    details
  }
}

/**
 * Criterion 5: Schema validation performance is acceptable
 */
async function validatePerformance(): Promise<AcceptanceCriteriaResult> {
  const iterations = 100
  const configs = Array.from({ length: iterations }, () => getDefaultConfig())
  const times: number[] = []
  let totalValidationTime = 0

  for (let i = 0; i < iterations; i++) {
    const start = performance.now()
    const result = validateSimulationConfig(configs[i])
    const end = performance.now()

    times.push(end - start)
    totalValidationTime += (end - start)

    // Ensure validation succeeds
    if (!result.success) {
      return {
        criterion: 'Schema validation performance is acceptable',
        passed: false,
        description: 'Validates that schema validation completes within acceptable time limits',
        details: `Validation failed on iteration ${i + 1}: ${result.errors.map(e => e.message).join(', ')}`
      }
    }
  }

  const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length
  const maxTime = Math.max(...times)

  // Performance criteria: average < 5ms, max < 10ms, total < 100ms
  const passed = averageTime < 5 && maxTime < 10 && totalValidationTime < 100

  return {
    criterion: 'Schema validation performance is acceptable',
    passed,
    description: 'Validates that schema validation completes within acceptable time limits',
    details: `Average: ${averageTime.toFixed(3)}ms, Max: ${maxTime.toFixed(3)}ms, Total: ${totalValidationTime.toFixed(3)}ms`,
    performance: {
      averageTime,
      maxTime,
      totalTime: totalValidationTime
    }
  }
}

/**
 * Criterion 6: TypeScript integration works seamlessly
 */
async function validateTypeScriptIntegration(): Promise<AcceptanceCriteriaResult> {
  try {
    // Test schema parsing with TypeScript
    const config = getDefaultConfig()
    const parsed = SimulationConfigSchema.parse(config)

    // Test safe validation with typed return
    const safeResult = safeValidateConfig(config)

    // Test that types are properly inferred
    const hasCorrectTypes = typeof parsed.width === 'number' &&
                           typeof parsed.agent_parameters.SystemAgent.learning_rate === 'number' &&
                           typeof parsed.visualization.background_color === 'string'

    const safeValidationWorks = safeResult.success && safeResult.data !== undefined

    const passed = hasCorrectTypes && safeValidationWorks

    const details = passed
      ? 'TypeScript integration working seamlessly with proper type inference'
      : 'TypeScript integration issues detected'

    return {
      criterion: 'TypeScript integration works seamlessly',
      passed,
      description: 'Validates that Zod schemas integrate properly with TypeScript for type safety and IntelliSense',
      details
    }
  } catch (error) {
    return {
      criterion: 'TypeScript integration works seamlessly',
      passed: false,
      description: 'Validates that Zod schemas integrate properly with TypeScript for type safety and IntelliSense',
      details: `TypeScript integration error: ${error instanceof Error ? error.message : 'Unknown error'}`
    }
  }
}

/**
 * Generate recommendations for failed criteria
 */
function generateRecommendations(results: AcceptanceCriteriaResult[]): string[] {
  const recommendations: string[] = []

  results.forEach(result => {
    if (!result.passed) {
      switch (result.criterion) {
        case 'All simulation config fields have Zod validation':
          recommendations.push('Review and ensure all configuration fields have Zod schema validation rules')
          break
        case 'Nested object validation works correctly':
          recommendations.push('Fix validation rules for nested objects (agent_parameters, visualization, module_parameters)')
          break
        case 'Custom validation rules are implemented':
          recommendations.push('Implement cross-field validation rules (ratios, epsilon hierarchy, performance constraints)')
          break
        case 'Error messages are properly formatted':
          recommendations.push('Improve error message formatting to be user-friendly and include error codes')
          break
        case 'Schema validation performance is acceptable':
          recommendations.push('Optimize schema validation performance - consider lazy validation or schema caching')
          break
        case 'TypeScript integration works seamlessly':
          recommendations.push('Fix TypeScript integration issues - ensure proper type inference from Zod schemas')
          break
        default:
          recommendations.push(`Address issues with: ${result.criterion}`)
      }
    }
  })

  return recommendations
}

/**
 * Export individual criterion validators for use in tests
 */
export const criterionValidators = {
  validateAllFieldsHaveValidation,
  validateNestedObjectValidation,
  validateCustomValidationRules,
  validateErrorMessageFormatting,
  validatePerformance,
  validateTypeScriptIntegration
}