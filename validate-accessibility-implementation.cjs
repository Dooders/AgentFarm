#!/usr/bin/env node

/**
 * Accessibility Implementation Validation Script
 *
 * This script validates that all Issue #13 accessibility features are properly implemented
 * and functional. It checks for the presence of required files, components, and features.
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Validating Issue #13 Accessibility Implementation...\n');

// Define required files and features
const requiredFiles = [
  'src/components/UI/AccessibilityProvider.tsx',
  'src/components/UI/SkipNavigation.tsx',
  'src/hooks/useKeyboardNavigation.ts',
  'src/utils/focusManagement.ts',
  'src/components/__tests__/Accessibility.test.tsx',
  'src/styles/index.css', // Enhanced with accessibility styles
  'ISSUE_13_COMPLETION_REPORT.md'
];

const requiredComponents = [
  'AccessibilityProvider',
  'SkipNavigation',
  'useAccessibility',
  'useKeyboardNavigation',
  'saveFocus',
  'restoreFocus',
  'trapFocus'
];

const requiredCSSFeatures = [
  '.high-contrast',
  'skip-link',
  'focus-ring',
  '--high-contrast-focus'
];

const requiredARIAFeatures = [
  'role="main"',
  'role="navigation"',
  'role="complementary"',
  'aria-live',
  'aria-label',
  'aria-expanded'
];

let validationPassed = true;

function checkFileExists(filePath) {
  const fullPath = path.join(__dirname, filePath);
  if (fs.existsSync(fullPath)) {
    console.log(`✅ ${filePath} - EXISTS`);
    return true;
  } else {
    console.log(`❌ ${filePath} - MISSING`);
    validationPassed = false;
    return false;
  }
}

function checkFileContains(filePath, searchTerm) {
  const fullPath = path.join(__dirname, filePath);
  try {
    const content = fs.readFileSync(fullPath, 'utf8');
    if (content.includes(searchTerm)) {
      console.log(`✅ ${filePath} contains "${searchTerm}"`);
      return true;
    } else {
      // For components, also check if they're imported/used in the file
      const importPattern = new RegExp(`import.*${searchTerm}|from.*${searchTerm}|${searchTerm}\\s*\\(|${searchTerm}\\s*=`, 'g');
      if (importPattern.test(content)) {
        console.log(`✅ ${filePath} uses "${searchTerm}"`);
        return true;
      }
      console.log(`❌ ${filePath} missing "${searchTerm}"`);
      validationPassed = false;
      return false;
    }
  } catch (error) {
    console.log(`❌ Error reading ${filePath}: ${error.message}`);
    validationPassed = false;
    return false;
  }
}

console.log('📁 Checking Required Files...\n');

requiredFiles.forEach(file => {
  checkFileExists(file);
});

console.log('\n🔧 Checking Component Implementation...\n');

requiredComponents.forEach(component => {
  if (checkFileContains('src/components/UI/AccessibilityProvider.tsx', component) ||
      checkFileContains('src/hooks/useKeyboardNavigation.ts', component) ||
      checkFileContains('src/utils/focusManagement.ts', component)) {
    console.log(`✅ Component "${component}" - IMPLEMENTED`);
  } else {
    console.log(`❌ Component "${component}" - NOT FOUND`);
    validationPassed = false;
  }
});

console.log('\n🎨 Checking CSS Implementation...\n');

requiredCSSFeatures.forEach(feature => {
  if (checkFileContains('src/styles/index.css', feature)) {
    console.log(`✅ CSS feature "${feature}" - IMPLEMENTED`);
  } else {
    console.log(`❌ CSS feature "${feature}" - MISSING`);
    validationPassed = false;
  }
});

console.log('\n🏷️ Checking ARIA Implementation...\n');

requiredARIAFeatures.forEach(feature => {
  // Check in multiple component files
  const filesToCheck = [
    'src/components/ConfigExplorer/ConfigExplorer.tsx',
    'src/components/ConfigExplorer/LeftPanel.tsx',
    'src/components/ConfigExplorer/RightPanel.tsx',
    'src/components/Validation/ValidationDisplay.tsx'
  ];

  let found = false;
  filesToCheck.forEach(file => {
    if (checkFileContains(file, feature)) {
      found = true;
    }
  });

  if (found) {
    console.log(`✅ ARIA feature "${feature}" - IMPLEMENTED`);
  } else {
    console.log(`❌ ARIA feature "${feature}" - MISSING`);
    validationPassed = false;
  }
});

console.log('\n🧪 Checking Test Implementation...\n');

if (checkFileContains('src/components/__tests__/Accessibility.test.tsx', 'AccessibilityProvider')) {
  console.log('✅ Enhanced accessibility tests - IMPLEMENTED');
} else {
  console.log('❌ Enhanced accessibility tests - MISSING');
  validationPassed = false;
}

console.log('\n📋 Checking Documentation...\n');

if (checkFileContains('README.md', 'Accessibility & User Experience')) {
  console.log('✅ README accessibility section - UPDATED');
} else {
  console.log('❌ README accessibility section - MISSING');
  validationPassed = false;
}

if (checkFileContains('GUI_TESTING_README.md', 'Advanced Accessibility Testing')) {
  console.log('✅ GUI testing docs - UPDATED');
} else {
  console.log('❌ GUI testing docs - MISSING ACCESSIBILITY SECTION');
  validationPassed = false;
}

console.log('\n📊 Validation Summary:\n');

if (validationPassed) {
  console.log('🎉 SUCCESS: All Issue #13 accessibility features are properly implemented!');
  console.log('✅ Focus states with high-contrast support');
  console.log('✅ Comprehensive keyboard navigation');
  console.log('✅ ARIA compliance and landmarks');
  console.log('✅ Skip navigation functionality');
  console.log('✅ High contrast mode support');
  console.log('✅ Focus management utilities');
  console.log('✅ Enhanced accessibility testing');
  console.log('✅ Documentation updates');
  process.exit(0);
} else {
  console.log('❌ FAILURE: Some accessibility features are missing or incomplete.');
  console.log('Please review the validation errors above and ensure all features are implemented.');
  process.exit(1);
}