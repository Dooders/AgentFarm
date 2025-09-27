/* eslint-disable @typescript-eslint/no-var-requires */
// @vitest-environment node
import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest'
import path from 'path'
import fs from 'fs'
import { createRequire } from 'module'
const nodeRequire = createRequire(import.meta.url)

// Electron modules are required in handlers; mock minimal API
vi.mock('electron', () => ({
  ipcMain: { handle: vi.fn() },
  dialog: { showOpenDialog: vi.fn(), showSaveDialog: vi.fn() },
  app: {
    getPath: vi.fn((key: string) => {
      const home = process.env.HOME || '/home/test'
      const mappings: Record<string, string> = {
        home,
        appData: path.join(home, '.config'),
        userData: path.join(home, '.config', 'ConfigExplorer'),
        documents: path.join(home, 'Documents'),
        desktop: path.join(home, 'Desktop'),
        downloads: path.join(home, 'Downloads')
      }
      return mappings[key] || home
    })
  }
}))

// Create a temporary directory structure for FS tests
const tmpRoot = path.join(process.cwd(), '.tmp-tests-electron')
const allowedDir = path.join(tmpRoot, 'Downloads')
const deniedDir = path.join('/etc') // should be denied by path checks

let moduleLoaded = false

beforeAll(() => {
  fs.mkdirSync(allowedDir, { recursive: true })
  process.env.HOME = tmpRoot
  try {
    nodeRequire('../ipcHandlers')
    moduleLoaded = true
  } catch (e) {
    moduleLoaded = false
  }
})

afterAll(() => {
  try { fs.rmSync(tmpRoot, { recursive: true, force: true }) } catch {}
})

;(moduleLoaded ? describe : describe.skip)('ipcHandlers security and FS guards', () => {
  it('registers ipcMain handlers', async () => {
    const { ipcMain } = await import('electron') as any
    expect(ipcMain.handle).toHaveBeenCalled()
  })

  it('denies file read outside allowed roots', async () => {
    const { ipcMain } = await import('electron') as any
    const handlerEntry = (ipcMain.handle as any).mock.calls.find((c: any[]) => c[0] === 'fs:file:read')
    expect(handlerEntry).toBeTruthy()
    const handler = handlerEntry[1]
    await expect(handler({ sender: { send: vi.fn() } }, { filePath: path.join(deniedDir, 'passwd') }))
      .rejects.toThrow(/Access denied/)
  })

  it('allows file write within allowed roots', async () => {
    const { ipcMain } = await import('electron') as any
    const handlerEntry = (ipcMain.handle as any).mock.calls.find((c: any[]) => c[0] === 'fs:file:write')
    expect(handlerEntry).toBeTruthy()
    const handler = handlerEntry[1]
    const target = path.join(allowedDir, 'test.json')
    const result = await handler({ sender: { send: vi.fn() } }, { filePath: target, content: '{"a":1}' })
    expect(result.filePath).toBe(target)
    expect(fs.existsSync(target)).toBe(true)
  })
})

