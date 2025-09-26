import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  // Load environment variables
  const env = loadEnv(mode, process.cwd(), '')

  const isElectron = env.IS_ELECTRON === 'true'
  const isDevelopment = mode === 'development'

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@components': path.resolve(__dirname, './src/components'),
        '@stores': path.resolve(__dirname, './src/stores'),
        '@services': path.resolve(__dirname, './src/services'),
        '@types': path.resolve(__dirname, './src/types'),
        '@hooks': path.resolve(__dirname, './src/hooks'),
        '@styles': path.resolve(__dirname, './src/styles')
      }
    },
    base: './',
    build: {
      outDir: 'dist',
      sourcemap: isDevelopment, // Only generate sourcemaps in development
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom'],
            ui: ['leva', 'zustand'],
            utils: ['zod', 'styled-components', 'polished']
          }
        }
      },
      // Optimize for Electron
      target: 'esnext',
      minify: isDevelopment ? false : 'esbuild',
      cssCodeSplit: true
    },
    server: {
      port: parseInt(env.VITE_DEV_SERVER_PORT) || 3000,
      host: env.VITE_DEV_SERVER_HOST || 'localhost',
      strictPort: false,
      cors: true,
      // Enable HMR for Electron
      hmr: {
        port: 24678,
        host: env.VITE_DEV_SERVER_HOST || 'localhost'
      },
      // Proxy API calls in development
      proxy: {
        '/api': {
          target: env.VITE_API_BASE_URL || 'http://localhost:3000',
          changeOrigin: true
        }
      }
    },
    // Environment variables
    define: {
      __ELECTRON__: isElectron,
      __DEV__: isDevelopment,
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
      // Make environment variables available at runtime
      'process.env.VITE_APP_NAME': JSON.stringify(env.VITE_APP_NAME),
      'process.env.VITE_APP_VERSION': JSON.stringify(env.VITE_APP_VERSION),
      'process.env.VITE_API_BASE_URL': JSON.stringify(env.VITE_API_BASE_URL),
      'process.env.VITE_DEBUG': JSON.stringify(env.VITE_DEBUG),
      'process.env.VITE_LOG_LEVEL': JSON.stringify(env.VITE_LOG_LEVEL)
    },
    // Optimize dependencies
    optimizeDeps: {
      include: ['react', 'react-dom', 'zustand', 'leva', 'zod', 'styled-components'],
      exclude: ['electron']
    },
    // Development tools
    esbuild: {
      drop: isDevelopment ? [] : ['console', 'debugger'],
      legalComments: 'none'
    }
  }
})