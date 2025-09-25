(function() {
    const { ipcRenderer } = require('electron')

    async function openConfigDialog() {
        try {
            const res = await ipcRenderer.invoke('dialog:openConfig')
            return res || { canceled: true }
        } catch (e) {
            return { canceled: true, error: String(e && e.message ? e.message : e) }
        }
    }

    async function saveConfigDialog(suggestedPath) {
        try {
            const res = await ipcRenderer.invoke('dialog:saveConfig', suggestedPath)
            return res || { canceled: true }
        } catch (e) {
            return { canceled: true, error: String(e && e.message ? e.message : e) }
        }
    }

    window.dialogService = {
        openConfigDialog,
        saveConfigDialog,
    }
})()

