import { z } from 'zod'

// Agent parameter schema with comprehensive validation
export const AgentParameterSchema = z.object({
  target_update_freq: z.number()
    .min(1, 'Target update frequency must be at least 1')
    .max(1000, 'Target update frequency must be at most 1000')
    .int('Target update frequency must be an integer'),

  memory_size: z.number()
    .min(1000, 'Memory size must be at least 1000')
    .max(1000000, 'Memory size must be at most 1,000,000')
    .int('Memory size must be an integer'),

  learning_rate: z.number()
    .min(0.0001, 'Learning rate must be at least 0.0001')
    .max(1.0, 'Learning rate must be at most 1.0')
    .positive('Learning rate must be positive'),

  gamma: z.number()
    .min(0.0, 'Gamma must be at least 0.0')
    .max(1.0, 'Gamma must be at most 1.0'),

  epsilon_start: z.number()
    .min(0.0, 'Epsilon start must be at least 0.0')
    .max(1.0, 'Epsilon start must be at most 1.0'),

  epsilon_min: z.number()
    .min(0.0, 'Epsilon minimum must be at least 0.0')
    .max(1.0, 'Epsilon minimum must be at most 1.0'),

  epsilon_decay: z.number()
    .min(0.9, 'Epsilon decay must be at least 0.9')
    .max(0.9999, 'Epsilon decay must be at most 0.9999'),

  dqn_hidden_size: z.number()
    .min(32, 'DQN hidden size must be at least 32')
    .max(2048, 'DQN hidden size must be at most 2048')
    .int('DQN hidden size must be an integer'),

  batch_size: z.number()
    .min(16, 'Batch size must be at least 16')
    .max(1024, 'Batch size must be at most 1024')
    .int('Batch size must be an integer'),

  tau: z.number()
    .min(0.001, 'Tau must be at least 0.001')
    .max(1.0, 'Tau must be at most 1.0'),

  success_reward: z.number()
    .min(0.1, 'Success reward must be at least 0.1')
    .max(100.0, 'Success reward must be at most 100.0'),

  failure_penalty: z.number()
    .min(0.1, 'Failure penalty must be at least 0.1')
    .max(100.0, 'Failure penalty must be at most 100.0'),

  base_cost: z.number()
    .min(0.0, 'Base cost must be at least 0.0')
    .max(10.0, 'Base cost must be at most 10.0')
}).refine((data) => data.epsilon_min <= data.epsilon_start, {
  message: 'Epsilon minimum must be less than or equal to epsilon start',
  path: ['epsilon_min']
}).refine((data) => data.gamma > 0, {
  message: 'Gamma must be greater than 0 for effective learning',
  path: ['gamma']
})

// Module parameter schema (similar to agent parameters but for modules)
export const ModuleParameterSchema = AgentParameterSchema.extend({
  // Module-specific refinements can be added here
}).refine((data) => data.batch_size <= 256, {
  message: 'Module batch size should not exceed 256 for performance',
  path: ['batch_size']
})

// Visualization configuration schema
export const VisualizationConfigSchema = z.object({
  canvas_width: z.number()
    .min(400, 'Canvas width must be at least 400')
    .max(1920, 'Canvas width must be at most 1920')
    .int('Canvas width must be an integer'),

  canvas_height: z.number()
    .min(300, 'Canvas height must be at least 300')
    .max(1080, 'Canvas height must be at most 1080')
    .int('Canvas height must be an integer'),

  background_color: z.string()
    .regex(/^#[0-9A-Fa-f]{6}$/, 'Background color must be a valid hex color code'),

  agent_colors: z.object({
    SystemAgent: z.string()
      .regex(/^#[0-9A-Fa-f]{6}$/, 'SystemAgent color must be a valid hex color code'),
    IndependentAgent: z.string()
      .regex(/^#[0-9A-Fa-f]{6}$/, 'IndependentAgent color must be a valid hex color code'),
    ControlAgent: z.string()
      .regex(/^#[0-9A-Fa-f]{6}$/, 'ControlAgent color must be a valid hex color code')
  }),

  show_metrics: z.boolean(),

  font_size: z.number()
    .min(8, 'Font size must be at least 8')
    .max(24, 'Font size must be at most 24')
    .int('Font size must be an integer'),

  line_width: z.number()
    .min(1, 'Line width must be at least 1')
    .max(10, 'Line width must be at most 10')
    .int('Line width must be an integer')
})

// Agent type ratios schema with sum validation
export const AgentTypeRatiosSchema = z.object({
  SystemAgent: z.number()
    .min(0.0, 'SystemAgent ratio must be at least 0.0')
    .max(1.0, 'SystemAgent ratio must be at most 1.0'),
  IndependentAgent: z.number()
    .min(0.0, 'IndependentAgent ratio must be at least 0.0')
    .max(1.0, 'IndependentAgent ratio must be at most 1.0'),
  ControlAgent: z.number()
    .min(0.0, 'ControlAgent ratio must be at least 0.0')
    .max(1.0, 'ControlAgent ratio must be at most 1.0')
}).refine((data) => {
  const sum = data.SystemAgent + data.IndependentAgent + data.ControlAgent
  return Math.abs(sum - 1.0) < 0.001
}, {
  message: 'Agent type ratios must sum to exactly 1.0',
  path: ['SystemAgent']
})

// Main simulation configuration schema
export const SimulationConfigSchema = z.object({
  // Environment settings
  width: z.number()
    .min(10, 'Width must be at least 10')
    .max(1000, 'Width must be at most 1000')
    .int('Width must be an integer'),

  height: z.number()
    .min(10, 'Height must be at least 10')
    .max(1000, 'Height must be at most 1000')
    .int('Height must be an integer'),

  position_discretization_method: z.enum(['floor', 'round', 'ceil'], {
    errorMap: () => ({ message: 'Position discretization method must be floor, round, or ceil' })
  }),

  use_bilinear_interpolation: z.boolean(),

  // Agent settings
  system_agents: z.number()
    .min(0, 'System agents must be at least 0')
    .max(10000, 'System agents must be at most 10,000')
    .int('System agents must be an integer'),

  independent_agents: z.number()
    .min(0, 'Independent agents must be at least 0')
    .max(10000, 'Independent agents must be at most 10,000')
    .int('Independent agents must be an integer'),

  control_agents: z.number()
    .min(0, 'Control agents must be at least 0')
    .max(10000, 'Control agents must be at most 10,000')
    .int('Control agents must be an integer'),

  agent_type_ratios: AgentTypeRatiosSchema,

  // Learning parameters
  learning_rate: z.number()
    .min(0.0001, 'Learning rate must be at least 0.0001')
    .max(1.0, 'Learning rate must be at most 1.0')
    .positive('Learning rate must be positive'),

  epsilon_start: z.number()
    .min(0.0, 'Epsilon start must be at least 0.0')
    .max(1.0, 'Epsilon start must be at most 1.0'),

  epsilon_min: z.number()
    .min(0.0, 'Epsilon minimum must be at least 0.0')
    .max(1.0, 'Epsilon minimum must be at most 1.0'),

  epsilon_decay: z.number()
    .min(0.9, 'Epsilon decay must be at least 0.9')
    .max(0.9999, 'Epsilon decay must be at most 0.9999'),

  // Agent parameters
  agent_parameters: z.object({
    SystemAgent: AgentParameterSchema,
    IndependentAgent: AgentParameterSchema,
    ControlAgent: AgentParameterSchema
  }),

  // Visualization
  visualization: VisualizationConfigSchema,

  // Module parameters
  gather_parameters: ModuleParameterSchema,
  share_parameters: ModuleParameterSchema,
  move_parameters: ModuleParameterSchema,
  attack_parameters: ModuleParameterSchema
}).refine((data) => data.epsilon_min <= data.epsilon_start, {
  message: 'Epsilon minimum must be less than or equal to epsilon start',
  path: ['epsilon_min']
}).refine((data) => {
  // Performance warning for large environments
  const totalAgents = data.system_agents + data.independent_agents + data.control_agents
  const environmentSize = data.width * data.height

  return totalAgents * environmentSize <= 50000000
}, {
  message: 'Large combination of agents and environment size may cause performance issues',
  path: ['width']
}).refine((data) => {
  // Validate that total agents don't exceed environment capacity
  const totalAgents = data.system_agents + data.independent_agents + data.control_agents
  const environmentCapacity = data.width * data.height

  return totalAgents <= environmentCapacity
}, {
  message: 'Total number of agents exceeds environment capacity',
  path: ['system_agents']
})

// Export types inferred from schemas
export type SimulationConfigType = z.infer<typeof SimulationConfigSchema>
export type AgentParameterType = z.infer<typeof AgentParameterSchema>
export type ModuleParameterType = z.infer<typeof ModuleParameterSchema>
export type VisualizationConfigType = z.infer<typeof VisualizationConfigSchema>
export type AgentTypeRatiosType = z.infer<typeof AgentTypeRatiosSchema>

// Export all schemas
export const schemas = {
  SimulationConfigSchema,
  AgentParameterSchema,
  ModuleParameterSchema,
  VisualizationConfigSchema,
  AgentTypeRatiosSchema
} as const