export const MODULE_NAME_MAPPING: Record<string, string> = {
  'Movement': 'move_parameters',
  'Gathering': 'gather_parameters',
  'Combat': 'attack_parameters',
  'Sharing': 'share_parameters'
}

export type ModuleDisplayName = keyof typeof MODULE_NAME_MAPPING

