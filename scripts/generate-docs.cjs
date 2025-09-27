#!/usr/bin/env node

const fs = require('fs')
const path = require('path')

/**
 * Generate project documentation
 */
function generateDocumentation() {
  try {
    const packageJson = require('../package.json')
    const docs = {
      name: packageJson.name,
      version: packageJson.version,
      description: packageJson.description,
      scripts: packageJson.scripts,
      dependencies: packageJson.dependencies,
      devDependencies: packageJson.devDependencies,
      build: {
        outDir: 'dist',
        sourcemap: true,
        target: 'esnext'
      }
    }

    // Write documentation to file
    const docsPath = path.join(__dirname, '..', 'PROJECT_DOCS.json')
    fs.writeFileSync(docsPath, JSON.stringify(docs, null, 2))

  console.log('📚 Project documentation generated successfully!')
  console.log(`📄 Documentation saved to: ${docsPath}`)
  } catch (error) {
    console.error('❌ Error generating documentation:', error.message)
    process.exit(1)
  }

  // Generate README with script documentation
  generateScriptDocumentation()
}

/**
 * Generate script documentation for README
 */
function generateScriptDocumentation() {
  const packageJson = require('../package.json')
  const scripts = packageJson.scripts

  let scriptDocs = '# Available Scripts\n\n'

  Object.entries(scripts).forEach(([script, command]) => {
    scriptDocs += `## \`${script}\`\n\`\`\`bash\n${command}\n\`\`\`\n\n`
  })

  const readmePath = path.join(__dirname, '..', 'SCRIPTS_README.md')
  fs.writeFileSync(readmePath, scriptDocs)

  console.log('📜 Script documentation generated successfully!')
  console.log(`📄 Script documentation saved to: ${readmePath}`)
}

/**
 * Generate environment configuration documentation
 */
function generateEnvDocumentation() {
  const envExample = fs.readFileSync(path.join(__dirname, '..', '.env.example'), 'utf8')

  let envDocs = '# Environment Variables\n\n'
  envDocs += 'The following environment variables can be configured:\n\n'

  const lines = envExample.split('\n')
  let currentSection = ''

  lines.forEach(line => {
    if (line.startsWith('#')) {
      currentSection = line.replace('#', '').trim()
      envDocs += `## ${currentSection}\n\n`
    } else if (line.includes('=')) {
      const [key, value] = line.split('=')
      envDocs += `- \`${key}\`: ${value || 'Custom value'}\n`
    }
  })

  const envDocsPath = path.join(__dirname, '..', 'ENV_CONFIG_README.md')
  fs.writeFileSync(envDocsPath, envDocs)

  console.log('🔧 Environment configuration documentation generated successfully!')
  console.log(`📄 Environment documentation saved to: ${envDocsPath}`)
}

// Run documentation generation
if (require.main === module) {
  console.log('🚀 Generating project documentation...')
  generateDocumentation()
  generateEnvDocumentation()
  console.log('✅ Documentation generation completed!')
}

module.exports = {
  generateDocumentation,
  generateScriptDocumentation,
  generateEnvDocumentation
}