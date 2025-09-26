import { ConfigExplorer } from '@/components/ConfigExplorer/ConfigExplorer'
import { AccessibilityProvider } from '@/components/UI/AccessibilityProvider'

function App() {
  return (
    <AccessibilityProvider>
      <div className="App">
        <ConfigExplorer />
      </div>
    </AccessibilityProvider>
  )
}

export default App