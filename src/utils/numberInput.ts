/**
 * Returns true if the provided string represents an intermediate numeric input state
 * that should not yet be parsed to a number. Examples include '-', '+', trailing '.',
 * or scientific notation prefix like '1e' or '1e-' while the user is typing.
 */
export function isIntermediateNumericInput(inputValue: string): boolean {
  if (inputValue === '-' || inputValue === '+') return true
  if (inputValue.endsWith('.')) return true
  if (/e[+-]?$/i.test(inputValue)) return true
  return false
}

