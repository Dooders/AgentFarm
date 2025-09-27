// Minimal shims to satisfy type resolution in lint environment

declare module 'styled-components' {
  const styled: any
  export function css(...args: any[]): any
  export default styled
}

declare module 'react/jsx-runtime' {
  export const jsx: any
  export const jsxs: any
  export const Fragment: any
}

declare module '@testing-library/react' {
  export const render: any
  export const screen: any
  export const fireEvent: any
}

