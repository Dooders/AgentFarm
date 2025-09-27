import React, { useRef, useEffect, useState } from 'react'
import styled from 'styled-components'
import { ValidationSummary } from '@/components/Validation/ValidationSummary'
import { useAccessibility } from '@/components/UI/AccessibilityProvider'
import { useConfigStore } from '@/stores/configStore'
import { configSelectors } from '@/stores/selectors'
import { ComparisonPanel } from './ComparisonPanel'
import { validationService } from '@/services/validationService'
import { PresetManager } from './PresetManager'
import { SearchResults } from '@/components/Search/SearchResults'
import { FilterControls } from '@/components/Search/FilterControls'
import { SearchBar } from '@/components/Search/SearchBar'
import { SavedSearches } from '@/components/Search/SavedSearches'
import { useSearchStore } from '@/stores/searchStore'

const RightPanelContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--background-primary);
`

const PanelHeader = styled.div`
  padding: 16px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--background-secondary);
`

const PanelTitle = styled.h2`
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
`

const ContentArea = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  background: var(--background-primary);
`

const SectionTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
`

const ContentSection = styled.div`
  margin-bottom: 24px;
  padding: 16px;
  background: var(--background-secondary);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-sm);
`

const ConfigItem = styled.div`
  margin-bottom: 12px;
  padding: 8px;
  background: var(--background-primary);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--accent-primary);
`

const ConfigLabel = styled.span`
  font-weight: 600;
  color: var(--text-primary);
`

const ConfigValue = styled.span`
  color: var(--text-secondary);
  margin-left: 8px;
  white-space: normal;
  word-break: break-word;
`

const ActionButton = styled.button`
  padding: 8px 16px;
  margin: 4px;
  background: var(--background-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;

  &:hover {
    background: var(--accent-primary);
    color: white;
  }
`

const Stats = styled.div`
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  color: var(--text-secondary);
  font-size: 12px;
  margin-top: 8px;
`

export const RightPanel: React.FC = React.memo(() => {
  const runSearch = useSearchStore((s) => s.runSearch)
  const isSearching = useSearchStore((s) => s.isSearching)
  const { announceToScreenReader } = useAccessibility()
  const showComparison = useConfigStore(configSelectors.getShowComparison)
  const compareConfig = useConfigStore(configSelectors.getCompareConfig)
  const comparisonFilePath = useConfigStore((s) => s.comparisonFilePath)
  const clearComparison = useConfigStore((s) => s.clearComparison)
  const toggleComparison = useConfigStore((s) => s.toggleComparison)
  const setComparison = useConfigStore((s) => s.setComparison)
  const setComparisonPath = useConfigStore((s) => s.setComparisonPath)
  const diffStats = useConfigStore(configSelectors.getComparisonStats)
  const applyAll = useConfigStore((s) => s.applyAllDifferencesFromComparison)

  // File input handling for loading comparison config (JSON for now)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const onChooseFile = () => fileInputRef.current?.click()
  const [comparisonError, setComparisonError] = useState<string | undefined>(undefined)
  const handleComparisonSelection = async (file: File) => {
    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      const result = await validationService.validateConfig(parsed)
      if (!result.success) {
        setComparison(null)
        setComparisonPath(undefined)
        setComparisonError('Invalid comparison file: schema validation failed')
        announceToScreenReader('Invalid comparison file', 'assertive')
      } else {
        setComparison(parsed)
        setComparisonPath(file.name)
        setComparisonError(undefined)
        announceToScreenReader('Comparison configuration loaded', 'polite')
      }
    } catch {
      setComparisonError('Invalid comparison file: unable to parse JSON')
      announceToScreenReader('Invalid comparison file', 'assertive')
    }
  }

  const onFileSelected: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      await handleComparisonSelection(file)
    } finally {
      // reset input to allow re-select same file
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  // Synchronized scrolling: mirror left panel scroll into comparison area
  useEffect(() => {
    const left = document.getElementById('left-scroll-area')
    const right = document.getElementById('comparison-scroll-area')
    if (!left || !right) return

    const onLeftScroll = () => {
      right.scrollTop = left.scrollTop
    }
    left.addEventListener('scroll', onLeftScroll)
    return () => {
      left.removeEventListener('scroll', onLeftScroll)
    }
  }, [showComparison])

  return (
    <RightPanelContainer role="complementary" aria-label="Configuration comparison and validation panel">
      <PanelHeader>
        <PanelTitle>Configuration Comparison</PanelTitle>
      </PanelHeader>

      <ContentArea id="comparison-content" tabIndex={-1}>
        <section aria-labelledby="search-panel-title" id="search-panel">
          <SectionTitle id="search-panel-title">Search</SectionTitle>
          <ContentSection>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
              <SearchBar />
              <ActionButton onClick={() => runSearch()} disabled={isSearching}>{isSearching ? 'Searching…' : 'Run'}</ActionButton>
            </div>
            <FilterControls />
            <div style={{ marginTop: 12 }}>
              <SearchResults />
            </div>
            <div style={{ marginTop: 16 }}>
              <SectionTitle>Saved Searches</SectionTitle>
              <SavedSearches />
            </div>
          </ContentSection>
        </section>
        <section aria-labelledby="current-config-title">
          <SectionTitle id="current-config-title">Current Configuration</SectionTitle>

          <ContentSection role="list" aria-label="Current configuration parameters">
            <ConfigItem role="listitem">
              <ConfigLabel>Environment Settings:</ConfigLabel>
              <ConfigValue>Grid: 100x100, Floor discretization</ConfigValue>
            </ConfigItem>
            <ConfigItem role="listitem">
              <ConfigLabel>Agent Counts:</ConfigLabel>
              <ConfigValue>System: 20, Independent: 20, Control: 10</ConfigValue>
            </ConfigItem>
            <ConfigItem role="listitem">
              <ConfigLabel>Learning Rate:</ConfigLabel>
              <ConfigValue>0.001 (Adaptive)</ConfigValue>
            </ConfigItem>
            <ConfigItem role="listitem">
              <ConfigLabel>Visualization:</ConfigLabel>
              <ConfigValue>Canvas: 800x600, Metrics: Enabled</ConfigValue>
            </ConfigItem>
          </ContentSection>
        </section>

        <section aria-labelledby="comparison-tools-title" id="comparison-tools">
          <SectionTitle id="comparison-tools-title">Comparison Tools</SectionTitle>

          <ContentSection role="group" aria-label="Configuration comparison actions">
            <div style={{ marginBottom: '16px' }}>
              <ActionButton onClick={() => { toggleComparison(); announceToScreenReader(`Comparison panel ${!showComparison ? 'shown' : 'hidden'}`, 'polite') }}>
                {showComparison ? 'Hide Comparison Panel' : 'Show Comparison Panel'}
              </ActionButton>
              <ActionButton onClick={onChooseFile} aria-describedby="compare-config-desc">
                Load Comparison Config
              </ActionButton>
              <ActionButton onClick={() => { clearComparison(); announceToScreenReader('Comparison cleared', 'polite') }} disabled={!compareConfig}>
                Clear Comparison
              </ActionButton>
              {compareConfig && (
                <ActionButton onClick={() => { applyAll(); announceToScreenReader('Applied all differences from comparison', 'polite') }} disabled={!compareConfig || (diffStats.added + diffStats.removed + diffStats.changed) === 0}>
                  Apply All Differences
                </ActionButton>
              )}
              <input ref={fileInputRef} type="file" accept="application/json,.json" style={{ display: 'none' }} onChange={onFileSelected} />
            </div>

            {compareConfig && (
              <Stats aria-label="Difference statistics">
                <span>Added: {diffStats.added}</span>
                <span>Removed: {diffStats.removed}</span>
                <span>Changed: {diffStats.changed}</span>
                <span>Unchanged: {diffStats.unchanged}</span>
                <span>Changed %: {diffStats.percentChanged}%</span>
              </Stats>
            )}

            {showComparison && (
              <div>
                <SectionTitle style={{ marginTop: '8px' }}>Comparison Panel {comparisonFilePath ? `(${comparisonFilePath})` : ''}</SectionTitle>
                <div id="comparison-scroll-area" style={{ maxHeight: '40vh', overflowY: 'auto' }}>
                  <ComparisonPanel compareConfig={compareConfig} comparisonFilePath={comparisonFilePath} errorMessage={comparisonError} />
                </div>
              </div>
            )}
          </ContentSection>
        </section>

        <section aria-labelledby="preset-manager-title" id="presets-section">
          <SectionTitle id="preset-manager-title">Presets</SectionTitle>
          <ContentSection role="region" aria-label="Preset manager">
            <PresetManager />
          </ContentSection>
        </section>

        <section aria-labelledby="validation-status-title">
          <SectionTitle id="validation-status-title">Validation Status</SectionTitle>

          <ContentSection id="validation-content" tabIndex={-1}>
            <ValidationSummary />
          </ContentSection>
        </section>
      </ContentArea>
    </RightPanelContainer>
  )
})