const { ipcMain, dialog, app } = require('electron')
const fs = require('fs').promises
const path = require('path')
const os = require('os')
const Store = require('electron-store')
const YAML = require('yaml')
const TOML = require('@iarna/toml')

// Initialize electron-store for configuration persistence
const store = new Store({
  name: 'config-explorer',
  defaults: {
    // Application settings
    settings: {
      theme: 'dark',
      language: 'en',
      autoSave: true,
      backupOnSave: true,
      maxHistoryEntries: 50
    },

    // Configuration templates
    templates: {
      system: {},
      user: {}
    },

    // Configuration history
    history: {
      entries: [],
      currentIndex: -1,
      maxEntries: 50
    },

    // UI state persistence
    ui: {
      panelSizes: { left: 300, right: 900 },
      lastOpenedFiles: [],
      windowBounds: null
    }
  }
})

// =====================================================
// Security: Allowed roots for file system access
// =====================================================

const allowedRoots = {
  home: app.getPath('home'),
  appData: app.getPath('appData'),
  userData: app.getPath('userData'),
  documents: app.getPath('documents'),
  desktop: app.getPath('desktop'),
  downloads: app.getPath('downloads')
}

function isPathWithin(root, target) {
  try {
    const resolvedRoot = path.resolve(root) + path.sep
    const resolvedTarget = path.resolve(target)
    return resolvedTarget.startsWith(resolvedRoot)
  } catch {
    return false
  }
}

function isPathAllowed(targetPath) {
  return Object.values(allowedRoots).some(root => isPathWithin(root, targetPath))
}

// =====================================================
// Helpers for format handling and recent files
// =====================================================

function getFormatFromPath(filePath) {
  const ext = path.extname(filePath || '').toLowerCase()
  if (ext === '.yml' || ext === '.yaml') return 'yaml'
  if (ext === '.toml') return 'toml'
  if (ext === '.xml') return 'xml'
  return 'json'
}

function parseByFormat(format, content) {
  switch ((format || 'json').toLowerCase()) {
    case 'yaml':
    case 'yml':
      return YAML.parse(content)
    case 'toml':
      return TOML.parse(content)
    case 'xml':
      // XML not supported yet; placeholder
      return JSON.parse(content)
    default:
      return JSON.parse(content)
  }
}

function serializeByFormat(format, obj) {
  switch ((format || 'json').toLowerCase()) {
    case 'yaml':
    case 'yml':
      return YAML.stringify(obj)
    case 'toml':
      return TOML.stringify(obj)
    case 'xml':
      // XML not supported yet; placeholder JSON
      return JSON.stringify(obj, null, 2)
    default:
      return JSON.stringify(obj, null, 2)
  }
}

async function addRecentFile(filePath, metadata = {}) {
  try {
    const last = store.get('ui.lastOpenedFiles', [])
    const without = last.filter(f => f.path !== filePath)
    const stats = await fs.stat(filePath).catch(() => null)
    const entry = {
      path: filePath,
      openedAt: Date.now(),
      size: stats?.size,
      modified: stats?.mtimeMs,
      format: getFormatFromPath(filePath),
      ...metadata
    }
    const next = [entry, ...without].slice(0, store.get('settings.maxHistoryEntries', 50))
    store.set('ui.lastOpenedFiles', next)
  } catch {}
}

// Config file operations
ipcMain.handle('config:load', async (event, request) => {
  try {
    const incoming = typeof request === 'string' ? { filePath: request } : (request || {})
    const fullPath = incoming.filePath || await getDefaultConfigPath()
    if (!isPathAllowed(fullPath)) {
      return { success: false, error: 'Access denied: path not allowed' }
    }
    const format = incoming.format || getFormatFromPath(fullPath)
    const content = await fs.readFile(fullPath, 'utf8')
    const config = parseByFormat(format, content)
    const stats = await fs.stat(fullPath).catch(() => null)

    await addRecentFile(fullPath, { action: 'load' })

    const result = {
      success: true,
      payload: {
        config,
        metadata: { format, size: stats?.size, modified: stats?.mtimeMs },
        filePath: fullPath,
        timestamp: Date.now()
      }
    }
    try { event.sender.send('config:loaded', { filePath: fullPath, metadata: { format }, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to load config:', error)
    return { success: false, error: `Failed to load configuration: ${error.message}` }
  }
})

ipcMain.handle('config:save', async (event, request) => {
  try {
    const { config, filePath, format, backup, ifMatchMtime } = request || {}
    const fullPath = filePath || await getDefaultConfigPath()
    if (!isPathAllowed(fullPath)) {
      return { success: false, error: 'Access denied: path not allowed' }
    }
    const targetFormat = format || getFormatFromPath(fullPath)

    // Conflict detection
    try {
      const stats = await fs.stat(fullPath)
      if (typeof ifMatchMtime === 'number' && stats.mtimeMs !== ifMatchMtime) {
        return {
          success: false,
          error: 'Write conflict: file modified by another process',
          conflict: true,
          currentMtime: stats.mtimeMs,
          filePath: fullPath
        }
      }
    } catch {}

    // Backup
    if (backup) {
      try {
        await fs.copyFile(fullPath, `${fullPath}.backup`)
      } catch {}
    }

    const content = serializeByFormat(targetFormat, config)
    await fs.writeFile(fullPath, content, 'utf8')
    const statsAfter = await fs.stat(fullPath)

    await addRecentFile(fullPath, { action: 'save' })

    const result = {
      success: true,
      payload: {
        filePath: fullPath,
        size: statsAfter.size,
        timestamp: Date.now(),
        backupCreated: !!backup,
        format: targetFormat,
        mtime: statsAfter.mtimeMs
      }
    }
    try { event.sender.send('config:saved', { filePath: fullPath, size: statsAfter.size, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to save config:', error)
    return { success: false, error: `Failed to save configuration: ${error.message}` }
  }
})

ipcMain.handle('config:export', async (event, request) => {
  try {
    const { config, format = 'json', filePath, includeMetadata = false, subsetPath, paths } = request || {}
    let outputConfig = config || {}
    if (subsetPath) {
      const parts = subsetPath.split('.')
      let cursor = outputConfig
      for (const k of parts) {
        if (cursor && Object.prototype.hasOwnProperty.call(cursor, k)) {
          cursor = cursor[k]
        } else {
          cursor = undefined
          break
        }
      }
      outputConfig = cursor
    } else if (Array.isArray(paths) && paths.length > 0) {
      const pick = {}
      for (const p of paths) {
        const parts = p.split('.')
        // verify path exists in source
        let verifySrc = outputConfig
        let exists = true
        for (const k of parts) {
          if (verifySrc && Object.prototype.hasOwnProperty.call(verifySrc, k)) {
            verifySrc = verifySrc[k]
          } else {
            exists = false
            break
          }
        }
        if (!exists) continue
        // build in destination
        let dst = pick
        let src = outputConfig
        for (let i = 0; i < parts.length; i++) {
          const k = parts[i]
          if (i === parts.length - 1) {
            dst[k] = src[k]
          } else {
            dst[k] = dst[k] || {}
            dst = dst[k]
            src = src[k]
          }
        }
      }
      outputConfig = pick
    }

    const content = serializeByFormat(format, includeMetadata ? { config: outputConfig, metadata: { exportedAt: Date.now() } } : outputConfig)

    let targetPath = filePath
    if (!targetPath) {
      const dialogResult = await dialog.showSaveDialog({
        title: 'Export Configuration',
        defaultPath: `config.${format}`,
        filters: [
          { name: 'JSON', extensions: ['json'] },
          { name: 'YAML', extensions: ['yaml', 'yml'] },
          { name: 'TOML', extensions: ['toml'] }
        ]
      })
      if (dialogResult.canceled) {
        return { cancelled: true }
      }
      targetPath = dialogResult.filePath
    }

    if (!isPathAllowed(targetPath)) {
      return { success: false, error: 'Access denied: path not allowed' }
    }

    await fs.writeFile(targetPath, content, 'utf8')
    const stats = await fs.stat(targetPath)

    const result = {
      success: true,
      payload: {
        filePath: targetPath,
        content,
        size: stats.size,
        format,
        timestamp: Date.now()
      }
    }
    try { event.sender.send('fs:operation:complete', { op: 'write', path: targetPath, size: stats.size, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to export config:', error)
    return { success: false, error: `Failed to export configuration: ${error.message}` }
  }
})

// Dialog operations
ipcMain.handle('dialog:open', async (event, options = {}) => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'Configuration Files', extensions: ['json', 'yaml', 'yml', 'toml'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    ...options
  })

  return result
})

ipcMain.handle('dialog:save', async (event, options = {}) => {
  const result = await dialog.showSaveDialog({
    filters: [
      { name: 'Configuration Files', extensions: ['json', 'yaml', 'yml', 'toml'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    ...options
  })

  return result
})

// =====================================================
// Enhanced Configuration Operations
// =====================================================

// Configuration import handler
ipcMain.handle('config:import', async (event, request) => {
  try {
    let configData

    if (request.content) {
      configData = request.content
    } else if (request.filePath) {
      configData = await fs.readFile(request.filePath, 'utf8')
    } else {
      throw new Error('Either content or filePath must be provided')
    }

    const format = request.format || getFormatFromPath(request.filePath || '')
    let config = parseByFormat(format, configData)

    if (request.validate !== false) {
      // Basic validation - in a real implementation, use the Zod schema
      if (!config || typeof config !== 'object') {
        throw new Error('Invalid configuration format')
      }
    }

    if (request.merge) {
      // Shallow merge for now
      const existingConfig = store.get('currentConfig', {})
      config = { ...existingConfig, ...config }
    }

    // Store the imported configuration
    store.set('currentConfig', config)

    const payload = {
      config,
      metadata: {
        importSource: request.filePath || 'content',
        importFormat: format,
        importedAt: new Date().toISOString()
      },
      source: request.filePath ? 'file' : 'content',
      timestamp: Date.now()
    }
    const result = { success: true, payload }
    try { event.sender.send('config:loaded', { source: payload.source, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to import config:', error)
    return { success: false, error: `Failed to import configuration: ${error.message}` }
  }
})

// Configuration validation handler
ipcMain.handle('config:validate', async (event, request) => {
  try {
    const startTime = Date.now()
    const errors = []
    const warnings = []

    // Basic validation
    if (!request.config || typeof request.config !== 'object') {
      errors.push('Configuration must be a valid object')
    }

    // Check for required fields based on actual config structure
    const requiredFields = ['width', 'height', 'system_agents', 'independent_agents', 'control_agents']
    for (const field of requiredFields) {
      if (request.config[field] === undefined || request.config[field] === null) {
        errors.push(`Missing required field: ${field}`)
      }
    }

    // Additional validation based on rules
    if (request.rules && Array.isArray(request.rules)) {
      for (const rule of request.rules) {
        switch (rule) {
          case 'population_positive':
            if (request.config.agents?.population && request.config.agents.population < 0) {
              errors.push('Population must be positive')
            }
            break
          case 'world_size_valid':
            if (request.config.environment?.world) {
              const { width, height } = request.config.environment.world
              if (width < 10 || height < 10 || width > 10000 || height > 10000) {
                warnings.push('World size should be between 10x10 and 10000x10000')
              }
            }
            break
        }
      }
    }

    const validationTime = Date.now() - startTime
    const isValid = errors.length === 0

    const result = {
      isValid,
      errors,
      warnings,
      config: request.config,
      validationTime,
      timestamp: Date.now()
    }
    try { event.sender.send('config:validation:complete', result) } catch {}
    if (!result.isValid) { try { event.sender.send('validation:error', { errors: result.errors, timestamp: Date.now() }) } catch {} }
    return result
  } catch (error) {
    console.error('Failed to validate config:', error)
    throw new Error(`Failed to validate configuration: ${error.message}`)
  }
})

// =====================================================
// Template Operations
// =====================================================

// Template load handler
ipcMain.handle('config:template:load', async (event, request) => {
  try {
    const { templateName, category = 'user' } = request
    const templates = store.get(`templates.${category}`, {})

    if (!templates[templateName]) {
      throw new Error(`Template '${templateName}' not found in category '${category}'`)
    }

    const template = templates[templateName]
    return {
      template: {
        name: templateName,
        description: template.description || '',
        category,
        author: template.author || 'user',
        version: template.version || '1.0.0',
        lastModified: template.lastModified || Date.now(),
        tags: template.tags || []
      },
      config: template.config,
      timestamp: Date.now()
    }
  } catch (error) {
    console.error('Failed to load template:', error)
    throw new Error(`Failed to load template: ${error.message}`)
  }
})

// Template save handler
ipcMain.handle('config:template:save', async (event, request) => {
  try {
    const { template, config, overwrite = false } = request
    const category = template.category || 'user'
    const templates = store.get(`templates.${category}`, {})

    if (!overwrite && templates[template.name]) {
      throw new Error(`Template '${template.name}' already exists. Use overwrite=true to replace.`)
    }

    templates[template.name] = {
      ...template,
      config,
      lastModified: Date.now(),
      created: templates[template.name]?.created || Date.now()
    }

    store.set(`templates.${category}`, templates)

    const result = {
      templateName: template.name,
      category,
      size: JSON.stringify(templates[template.name]).length,
      timestamp: Date.now(),
      created: !templates[template.name].created || overwrite,
      overwritten: overwrite
    }
    try { event.sender.send('config:template:created', { name: template.name, category, overwritten: overwrite, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to save template:', error)
    throw new Error(`Failed to save template: ${error.message}`)
  }
})

// Template delete handler
ipcMain.handle('config:template:delete', async (event, request) => {
  try {
    const { templateName, category = 'user' } = request
    const templates = store.get(`templates.${category}`, {})

    if (!templates[templateName]) {
      throw new Error(`Template '${templateName}' not found in category '${category}'`)
    }

    delete templates[templateName]
    store.set(`templates.${category}`, templates)

    const result = {
      templateName,
      category,
      timestamp: Date.now(),
      deleted: true
    }
    try { event.sender.send('config:template:deleted', { name: templateName, category, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to delete template:', error)
    throw new Error(`Failed to delete template: ${error.message}`)
  }
})

// Template list handler
ipcMain.handle('config:template:list', async (event, request) => {
  try {
    const { category, includeSystem = true, includeUser = true } = request
    const templates = []
    const categoryCount = {}

    if (includeUser) {
      const userTemplates = store.get('templates.user', {})
      Object.entries(userTemplates).forEach(([name, template]) => {
        templates.push({
          name,
          description: template.description || '',
          category: 'user',
          author: template.author || 'user',
          version: template.version || '1.0.0',
          lastModified: template.lastModified || Date.now(),
          size: JSON.stringify(template.config).length,
          type: 'user'
        })
      })
      categoryCount.user = Object.keys(userTemplates).length
    }

    if (includeSystem) {
      const systemTemplates = store.get('templates.system', {})
      Object.entries(systemTemplates).forEach(([name, template]) => {
        templates.push({
          name,
          description: template.description || '',
          category: 'system',
          author: template.author || 'system',
          version: template.version || '1.0.0',
          lastModified: template.lastModified || Date.now(),
          size: JSON.stringify(template.config).length,
          type: 'system'
        })
      })
      categoryCount.system = Object.keys(systemTemplates).length
    }

    return {
      templates,
      totalCount: templates.length,
      categoryCount,
      timestamp: Date.now()
    }
  } catch (error) {
    console.error('Failed to list templates:', error)
    throw new Error(`Failed to list templates: ${error.message}`)
  }
})

// =====================================================
// History Operations
// =====================================================

// History save handler
ipcMain.handle('config:history:save', async (event, request) => {
  try {
    const { history, currentIndex, maxEntries } = request
    const maxEntriesLimit = maxEntries || store.get('settings.maxHistoryEntries', 50)

    // Limit history size
    const limitedHistory = history.slice(-maxEntriesLimit)

    store.set('history.entries', limitedHistory)
    store.set('history.currentIndex', Math.min(currentIndex, limitedHistory.length - 1))

    const result = {
      entriesSaved: limitedHistory.length,
      currentIndex: Math.min(currentIndex, limitedHistory.length - 1),
      timestamp: Date.now()
    }
    try { event.sender.send('config:history:updated', { totalEntries: limitedHistory.length, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to save history:', error)
    throw new Error(`Failed to save history: ${error.message}`)
  }
})

// History load handler
ipcMain.handle('config:history:load', async (event, request) => {
  try {
    const { maxEntries, since, filter } = request
    let history = store.get('history.entries', [])
    let currentIndex = store.get('history.currentIndex', -1)

    // Apply filters
    if (since) {
      history = history.filter(entry => entry.timestamp >= since)
    }

    if (filter?.actions && filter.actions.length > 0) {
      history = history.filter(entry => filter.actions.includes(entry.action))
    }

    if (filter?.dateRange) {
      history = history.filter(entry =>
        entry.timestamp >= filter.dateRange.start &&
        entry.timestamp <= filter.dateRange.end
      )
    }

    if (maxEntries) {
      history = history.slice(-maxEntries)
      // Adjust current index if it's out of bounds
      if (currentIndex >= history.length) {
        currentIndex = history.length - 1
      }
    }

    const result = {
      history,
      currentIndex,
      totalEntries: history.length,
      timestamp: Date.now()
    }
    try { event.sender.send('config:history:updated', { totalEntries: history.length, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to load history:', error)
    throw new Error(`Failed to load history: ${error.message}`)
  }
})

// History clear handler
ipcMain.handle('config:history:clear', async (event) => {
  try {
    store.set('history.entries', [])
    store.set('history.currentIndex', -1)

    const result = {
      success: true,
      timestamp: Date.now()
    }
    try { event.sender.send('config:history:updated', { totalEntries: 0, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to clear history:', error)
    throw new Error(`Failed to clear history: ${error.message}`)
  }
})

// =====================================================
// Enhanced File System Operations
// =====================================================

// File exists handler
ipcMain.handle('fs:file:exists', async (event, request) => {
  try {
    const { filePath } = request
    if (!isPathAllowed(filePath)) {
      return {
        exists: false,
        isFile: false,
        isDirectory: false
      }
    }
    const stats = await fs.stat(filePath)

    return {
      exists: true,
      isFile: stats.isFile(),
      isDirectory: stats.isDirectory(),
      size: stats.size,
      modified: stats.mtimeMs
    }
  } catch (error) {
    if (error.code === 'ENOENT') {
      return {
        exists: false,
        isFile: false,
        isDirectory: false
      }
    }
    throw error
  }
})

// File read handler
ipcMain.handle('fs:file:read', async (event, request) => {
  try {
    const { filePath, encoding = 'utf8', options = {} } = request
    if (!isPathAllowed(filePath)) {
      throw new Error('Access denied: path not allowed')
    }
    const content = await fs.readFile(filePath, { encoding, ...options })

    const stats = await fs.stat(filePath)

    const result = {
      content,
      encoding,
      size: stats.size,
      modified: stats.mtimeMs,
      mimeType: getMimeType(filePath)
    }
    try { event.sender.send('fs:operation:complete', { op: 'read', path: filePath, size: stats.size, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to read file:', error)
    throw new Error(`Failed to read file: ${error.message}`)
  }
})

// File write handler
ipcMain.handle('fs:file:write', async (event, request) => {
  try {
    const { filePath, content, encoding = 'utf8', options = {} } = request
    if (!isPathAllowed(filePath)) {
      throw new Error('Access denied: path not allowed')
    }

    // Create backup if requested
    if (options.backup) {
      const backupPath = `${filePath}.backup`
      try {
        await fs.copyFile(filePath, backupPath)
        options.backupPath = backupPath
      } catch (error) {
        // File doesn't exist, no backup needed
      }
    }

    await fs.writeFile(filePath, content, { encoding, ...options })

    const stats = await fs.stat(filePath)

    const result = {
      filePath,
      written: content.length,
      size: stats.size,
      created: !options.backup,
      backupCreated: !!options.backup,
      backupPath: options.backupPath
    }
    try { event.sender.send('fs:operation:complete', { op: 'write', path: filePath, size: stats.size, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to write file:', error)
    throw new Error(`Failed to write file: ${error.message}`)
  }
})

// File delete handler
ipcMain.handle('fs:file:delete', async (event, request) => {
  try {
    const { filePath, backup = false } = request
    if (!isPathAllowed(filePath)) {
      throw new Error('Access denied: path not allowed')
    }

    if (backup) {
      const backupPath = `${filePath}.backup`
      await fs.copyFile(filePath, backupPath)
      request.backupPath = backupPath
    }

    await fs.unlink(filePath)

    const result = {
      filePath,
      deleted: true,
      backupCreated: backup,
      backupPath: request.backupPath
    }
    try { event.sender.send('fs:operation:complete', { op: 'delete', path: filePath, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to delete file:', error)
    throw new Error(`Failed to delete file: ${error.message}`)
  }
})

// Directory read handler
ipcMain.handle('fs:directory:read', async (event, request) => {
  try {
    const { dirPath, recursive = false, filter = {} } = request
    if (!isPathAllowed(dirPath)) {
      throw new Error('Access denied: path not allowed')
    }
    const entries = []

    const readDirectory = async (currentPath, relativePath = '') => {
      const items = await fs.readdir(currentPath, { withFileTypes: true })

      for (const item of items) {
        const fullPath = path.join(currentPath, item.name)
        const itemPath = path.join(relativePath, item.name)
        const stats = await fs.stat(fullPath)

        let includeItem = true

        // Apply filters
        if (filter.extensions && filter.extensions.length > 0) {
          const ext = path.extname(item.name)
          includeItem = filter.extensions.includes(ext)
        }

        if (filter.exclude && filter.exclude.includes(item.name)) {
          includeItem = false
        }

        if (filter.includeHidden === false && item.name.startsWith('.')) {
          includeItem = false
        }

        if (includeItem) {
          entries.push({
            name: item.name,
            path: itemPath,
            type: item.isDirectory() ? 'directory' : item.isFile() ? 'file' : 'symlink',
            size: item.isFile() ? stats.size : undefined,
            modified: stats.mtimeMs,
            extension: item.isFile() ? path.extname(item.name) : undefined
          })
        }

        if (recursive && item.isDirectory()) {
          await readDirectory(fullPath, itemPath)
        }
      }
    }

    await readDirectory(dirPath)

    const fileCount = entries.filter(e => e.type === 'file').length
    const directoryCount = entries.filter(e => e.type === 'directory').length

    const result = {
      entries,
      totalCount: entries.length,
      fileCount,
      directoryCount
    }
    try { event.sender.send('fs:operation:complete', { op: 'readDir', path: dirPath, totalCount: entries.length, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to read directory:', error)
    throw new Error(`Failed to read directory: ${error.message}`)
  }
})

// Directory create handler
ipcMain.handle('fs:directory:create', async (event, request) => {
  try {
    const { dirPath, recursive = true, mode = 0o755 } = request
    if (!isPathAllowed(dirPath)) {
      throw new Error('Access denied: path not allowed')
    }

    const exists = await fs.access(dirPath).then(() => true).catch(() => false)

    await fs.mkdir(dirPath, { recursive, mode })

    const result = {
      dirPath,
      created: !exists,
      existing: exists
    }
    try { event.sender.send('fs:operation:complete', { op: 'mkdir', path: dirPath, created: !exists, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to create directory:', error)
    throw new Error(`Failed to create directory: ${error.message}`)
  }
})

// Directory delete handler
ipcMain.handle('fs:directory:delete', async (event, request) => {
  try {
    const { dirPath, recursive = false, backup = false } = request
    if (!isPathAllowed(dirPath)) {
      throw new Error('Access denied: path not allowed')
    }

    if (backup) {
      const backupPath = `${dirPath}.backup`
      await fs.cp(dirPath, backupPath, { recursive: true })
      request.backupPath = backupPath
    }

    await fs.rm(dirPath, { recursive })

    const result = {
      dirPath,
      deleted: true,
      backupCreated: backup,
      backupPath: request.backupPath
    }
    try { event.sender.send('fs:operation:complete', { op: 'rmdir', path: dirPath, recursive, timestamp: Date.now() }) } catch {}
    return result
  } catch (error) {
    console.error('Failed to delete directory:', error)
    throw new Error(`Failed to delete directory: ${error.message}`)
  }
})

// =====================================================
// Application Operations
// =====================================================

// App ping handler for connection testing
ipcMain.handle('app:ping', async (event, request) => {
  return { success: true, timestamp: Date.now(), ...request }
})

// Settings get handler
ipcMain.handle('app:settings:get', async (event, request) => {
  try {
    const { keys, category = 'settings' } = request
    const settings = store.get(category, {})

    if (keys && keys.length > 0) {
      const filteredSettings = {}
      keys.forEach(key => {
        filteredSettings[key] = settings[key]
      })
      return { settings: filteredSettings, timestamp: Date.now() }
    }

    return { settings, timestamp: Date.now() }
  } catch (error) {
    console.error('Failed to get settings:', error)
    throw new Error(`Failed to get settings: ${error.message}`)
  }
})

// Settings set handler
ipcMain.handle('app:settings:set', async (event, request) => {
  try {
    const { settings, category = 'settings', persist = true } = request
    const currentSettings = store.get(category, {})

    const updatedSettings = { ...currentSettings, ...settings }
    const updatedKeys = Object.keys(settings)

    if (persist) {
      store.set(category, updatedSettings)
    }

    return {
      success: true,
      updatedKeys,
      timestamp: Date.now()
    }
  } catch (error) {
    console.error('Failed to set settings:', error)
    throw new Error(`Failed to set settings: ${error.message}`)
  }
})

// App version handler
ipcMain.handle('app:version:get', async (event) => {
  return {
    version: app.getVersion(),
    build: process.env.BUILD_NUMBER || 'dev',
    platform: process.platform,
    arch: process.arch,
    timestamp: Date.now()
  }
})

// App path handler
ipcMain.handle('app:path:get', async (event, request) => {
  try {
    const { type } = request

    const pathMap = {
      'home': app.getPath('home'),
      'appData': app.getPath('appData'),
      'userData': app.getPath('userData'),
      'cache': app.getPath('cache'),
      'temp': app.getPath('temp'),
      'exe': app.getPath('exe'),
      'module': app.getPath('module'),
      'desktop': app.getPath('desktop'),
      'documents': app.getPath('documents'),
      'downloads': app.getPath('downloads'),
      'music': app.getPath('music'),
      'pictures': app.getPath('pictures'),
      'videos': app.getPath('videos'),
      'logs': app.getPath('logs'),
      'crashDumps': app.getPath('crashDumps')
    }

    const resultPath = pathMap[type] || app.getPath('userData')

    return {
      path: resultPath,
      type,
      timestamp: Date.now()
    }
  } catch (error) {
    console.error('Failed to get app path:', error)
    throw new Error(`Failed to get app path: ${error.message}`)
  }
})

// System info handler
ipcMain.handle('system:info:get', async (event) => {
  const cpus = os.cpus()
  const totalMemory = os.totalmem()
  const freeMemory = os.freemem()

  return {
    platform: process.platform,
    arch: process.arch,
    release: os.release(),
    hostname: os.hostname(),
    userInfo: {
      uid: os.userInfo().uid,
      gid: os.userInfo().gid,
      username: os.userInfo().username,
      homedir: os.userInfo().homedir,
      shell: os.userInfo().shell
    },
    memory: {
      total: totalMemory,
      free: freeMemory,
      used: totalMemory - freeMemory
    },
    cpus: cpus.map(cpu => ({
      model: cpu.model,
      speed: cpu.speed,
      times: {
        user: cpu.times.user,
        nice: cpu.times.nice,
        sys: cpu.times.sys,
        idle: cpu.times.idle,
        irq: cpu.times.irq
      }
    })),
    loadavg: os.loadavg(),
    uptime: os.uptime(),
    timestamp: Date.now()
  }
})

// =====================================================
// Utility Functions
// =====================================================

// Helper function to get MIME type from file extension
function getMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  const mimeTypes = {
    '.json': 'application/json',
    '.yaml': 'application/x-yaml',
    '.yml': 'application/x-yaml',
    '.xml': 'application/xml',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.js': 'application/javascript',
    '.ts': 'application/typescript',
    '.html': 'text/html',
    '.css': 'text/css'
  }
  return mimeTypes[ext] || 'application/octet-stream'
}

// Helper functions
async function getDefaultConfigPath() {
  const appDataPath = process.platform === 'darwin'
    ? path.join(process.env.HOME, 'Library', 'Application Support', 'ConfigExplorer')
    : path.join(process.env.APPDATA || path.join(process.env.HOME, '.config'), 'ConfigExplorer')

  await fs.mkdir(appDataPath, { recursive: true })
  return path.join(appDataPath, 'default-config.json')
}