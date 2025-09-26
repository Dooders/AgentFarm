declare module 'msw' {
  export const rest: any
  export type RestRequest = any
  export type ResponseComposition<T = any> = any
  export type DefaultBodyType = any
  export type RestContext = any
  export function setupWorker(...handlers: any[]): any
}

declare module 'msw/node' {
  export function setupServer(...handlers: any[]): any
}
