// Type declarations for @testing-library/react
declare module '@testing-library/react' {
  export { renderHook, act } from '@testing-library/react/pure'
}

declare module '@testing-library/react/pure' {
  export function renderHook<TProps, TResult>(
    callback: (props: TProps) => TResult,
    options?: {
      initialProps?: TProps
      wrapper?: React.ComponentType<{ children: React.ReactNode }>
    }
  ): {
    result: {
      current: TResult
      error?: Error
    }
    rerender: (newProps?: TProps) => void
    unmount: () => void
  }

  export function act(callback: () => void | Promise<void>): Promise<void> | void
  export function act<T>(callback: () => T): T
}

