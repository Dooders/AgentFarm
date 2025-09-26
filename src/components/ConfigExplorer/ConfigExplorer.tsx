import React, { useEffect, useState } from 'react'
import { DualPanelLayout } from '@/components/Layout/DualPanelLayout'
import { ipcService } from '@/services/ipcService'
import { IPCConnectionStatus } from '@/types/ipc'

export const ConfigExplorer: React.FC = () => {
  const [connectionStatus, setConnectionStatus] = useState<IPCConnectionStatus>('disconnected')
  const [isInitialized, setIsInitialized] = useState(false)

  useEffect(() => {
    const initializeIPC = async () => {
      try {
        // Check if we're running in Electron
        if (typeof window !== 'undefined' && window.electronAPI) {
          console.log('Initializing IPC service...')

          // Wait for IPC service to initialize with timeout
          await new Promise<void>((resolve, reject) => {
            const maxAttempts = 50 // 5 seconds timeout
            let attempts = 0

            const checkConnection = () => {
              attempts++
              const status = ipcService.getConnectionStatus()

              if (status === 'connected') {
                resolve()
              } else if (attempts >= maxAttempts) {
                reject(new Error('IPC service connection timeout'))
              } else {
                setTimeout(checkConnection, 100)
              }
            }
            checkConnection()
          })

          setConnectionStatus('connected')
          console.log('IPC service initialized successfully')
        } else {
          console.log('Running in browser mode - IPC service not available')
          setConnectionStatus('disconnected')
        }
      } catch (error) {
        console.error('Failed to initialize IPC service:', error)
        setConnectionStatus('error')
      } finally {
        setIsInitialized(true)
      }
    }

    initializeIPC()

    // Set up connection status monitoring
    const statusCheckInterval = setInterval(() => {
      const currentStatus = ipcService.getConnectionStatus()
      setConnectionStatus(currentStatus)
    }, 1000)

    return () => {
      clearInterval(statusCheckInterval)
    }
  }, [])

  if (!isInitialized) {
    return (
      <div className="config-explorer loading">
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Initializing application...</p>
        </div>
      </div>
    )
  }

  if (connectionStatus === 'error') {
    return (
      <div className="config-explorer error">
        <div className="error-indicator">
          <h2>Connection Error</h2>
          <p>Failed to establish connection to backend services.</p>
          <button
            onClick={() => window.location.reload()}
            className="retry-button"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="config-explorer">
      {connectionStatus === 'disconnected' && (
        <div className="connection-status disconnected">
          <span>⚠️ Running in browser mode - some features may be limited</span>
        </div>
      )}
      <DualPanelLayout />
    </div>
  )
}