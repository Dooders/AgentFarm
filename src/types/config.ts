import { SimulationConfigType, AgentParameterType, VisualizationConfigType, AgentTypeRatiosType, ModuleParameterType } from './validation'

// Use Zod-inferred types for type safety
export type SimulationConfig = SimulationConfigType

// Use Zod-inferred types for type safety
export type AgentParameters = AgentParameterType
export type ModuleParameters = ModuleParameterType
export type VisualizationConfig = VisualizationConfigType

// Re-export types for convenience
export type {
  SimulationConfigType,
  AgentParameterType,
  ModuleParameterType,
  VisualizationConfigType,
  AgentTypeRatiosType
} from './validation'