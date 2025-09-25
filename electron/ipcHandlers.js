const { ipcMain, dialog } = require('electron')
const fs = require('fs').promises
const path = require('path')

// Config file operations
ipcMain.handle('config:load', async (event, filePath) => {
  try {
    const fullPath = filePath || await getDefaultConfigPath()
    const configData = await fs.readFile(fullPath, 'utf8')
    return JSON.parse(configData)
  } catch (error) {
    console.error('Failed to load config:', error)
    throw new Error(`Failed to load configuration: ${error.message}`)
  }
})

ipcMain.handle('config:save', async (event, config, filePath) => {
  try {
    const fullPath = filePath || await getDefaultConfigPath()
    await fs.writeFile(fullPath, JSON.stringify(config, null, 2))
    return { success: true, path: fullPath }
  } catch (error) {
    console.error('Failed to save config:', error)
    throw new Error(`Failed to save configuration: ${error.message}`)
  }
})

ipcMain.handle('config:export', async (event, config, format) => {
  try {
    const { filePath } = await dialog.showSaveDialog({
      title: 'Export Configuration',
      defaultPath: `config.${format}`,
      filters: [
        { name: 'JSON', extensions: ['json'] },
        { name: 'YAML', extensions: ['yaml', 'yml'] }
      ]
    })

    if (!filePath) return { success: false, cancelled: true }

    let content
    if (format === 'yaml') {
      // In a real implementation, you'd use a YAML library
      content = JSON.stringify(config, null, 2)
    } else {
      content = JSON.stringify(config, null, 2)
    }

    await fs.writeFile(filePath, content)
    return { success: true, path: filePath }
  } catch (error) {
    console.error('Failed to export config:', error)
    throw new Error(`Failed to export configuration: ${error.message}`)
  }
})

// Dialog operations
ipcMain.handle('dialog:open', async (event, options = {}) => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'Configuration Files', extensions: ['json', 'yaml', 'yml'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    ...options
  })

  return result
})

ipcMain.handle('dialog:save', async (event, options = {}) => {
  const result = await dialog.showSaveDialog({
    filters: [
      { name: 'Configuration Files', extensions: ['json', 'yaml', 'yml'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    ...options
  })

  return result
})

// Helper functions
async function getDefaultConfigPath() {
  const appDataPath = process.platform === 'darwin'
    ? path.join(process.env.HOME, 'Library', 'Application Support', 'ConfigExplorer')
    : path.join(process.env.APPDATA || path.join(process.env.HOME, '.config'), 'ConfigExplorer')

  await fs.mkdir(appDataPath, { recursive: true })
  return path.join(appDataPath, 'default-config.json')
}