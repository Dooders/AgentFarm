declare module 'msw' {
  export const http: any
  export type HttpRequest = any
  export type ResponseComposition<T = any> = any
  export type DefaultBodyType = any
  export type HttpContext = any
  export function setupWorker(...handlers: any[]): any
}

declare module 'msw/node' {
  export function setupServer(...handlers: any[]): any
}
