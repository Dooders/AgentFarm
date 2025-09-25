const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const path = require('path')
const { spawn } = require('child_process')

let pythonProcess = null

function startPythonBackend() {
    // Get the path to the Python executable
    const pythonExecutable = path.join(
        process.resourcesPath, 
        'python',
        process.platform === 'win32' ? 'simulation_backend.exe' : 'simulation_backend'
    )

    // Start the Python process
    pythonProcess = spawn(pythonExecutable)

    // Handle Python process output
    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python output: ${data}`)
    })

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python error: ${data}`)
    })

    // Wait for backend to start
    return new Promise((resolve) => {
        setTimeout(resolve, 2000)
    })
}

async function createWindow() {
    // Start Python backend first
    await startPythonBackend()

    // Create the browser window
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    })

    // Load the index.html file
    mainWindow.loadFile('editor/index.html')

    // Open DevTools in development
    if (process.env.NODE_ENV === 'development') {
        mainWindow.webContents.openDevTools()
    }
}

// Create window when app is ready
app.whenReady().then(createWindow)

// Quit when all windows are closed
app.on('window-all-closed', () => {
    // Kill Python process
    if (pythonProcess) {
        pythonProcess.kill()
    }
    
    if (process.platform !== 'darwin') {
        app.quit()
    }
})

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow()
    }
}) 

// IPC: Native File Dialogs for Config
ipcMain.handle('dialog:openConfig', async () => {
    const win = BrowserWindow.getFocusedWindow()
    const result = await dialog.showOpenDialog(win || undefined, {
        title: 'Open Configuration',
        properties: ['openFile'],
        filters: [
            { name: 'YAML', extensions: ['yml', 'yaml'] },
            { name: 'JSON', extensions: ['json'] },
            { name: 'All Files', extensions: ['*'] }
        ]
    })
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
        return { canceled: true }
    }
    return { canceled: false, filePath: result.filePaths[0] }
})

ipcMain.handle('dialog:saveConfig', async (_evt, suggestedPath) => {
    const win = BrowserWindow.getFocusedWindow()
    const result = await dialog.showSaveDialog(win || undefined, {
        title: 'Save Configuration',
        defaultPath: suggestedPath || undefined,
        filters: [
            { name: 'YAML', extensions: ['yml', 'yaml'] },
            { name: 'JSON', extensions: ['json'] }
        ]
    })
    if (result.canceled || !result.filePath) {
        return { canceled: true }
    }
    return { canceled: false, filePath: result.filePath }
})