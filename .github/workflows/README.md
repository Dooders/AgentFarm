# GitHub Actions Workflows

This directory contains comprehensive GitHub Actions workflows for the AgentFarm project, providing automated testing, validation, benchmarking, security scanning, and release management.

## Workflows Overview

### 1. CI/CD Pipeline (`ci.yml`)
**Triggers**: Push to main/develop, Pull Requests, Weekly schedule
**Purpose**: Comprehensive continuous integration and deployment pipeline

**Jobs:**
- **Code Quality & Linting**: Black, isort, Ruff, Pylint, MyPy
- **Testing & Coverage**: Multi-version Python testing with pytest and coverage
- **Security Scanning**: Safety, Bandit, Semgrep
- **Performance Benchmarking**: Memory DB, Perception Metrics, Observation Flow
- **Integration Testing**: Redis integration, simulation execution
- **Documentation & Build**: Documentation validation
- **Performance Regression Detection**: PR performance comparison
- **Dependency Management**: Conflict detection, outdated packages
- **Notifications**: Success/failure notifications

### 2. Comprehensive Benchmarking (`benchmark.yml`)
**Triggers**: Push to main, PRs, Daily schedule, Manual dispatch
**Purpose**: Detailed performance benchmarking and regression detection

**Jobs:**
- **Perception System Benchmarks**: Small and medium scale perception metrics
- **Memory System Benchmarks**: Memory DB, Redis, Pragma profiling
- **Performance Regression Testing**: Baseline vs current performance comparison
- **Stress Testing**: High-load simulation testing
- **Benchmark Results Analysis**: Automated result analysis and PR comments

### 3. Security Scanning (`security.yml`)
**Triggers**: Push to main/develop, PRs, Weekly schedule, Manual dispatch
**Purpose**: Comprehensive security vulnerability scanning

**Jobs:**
- **Dependency Security Scan**: Safety, pip-audit vulnerability checks
- **Static Security Analysis**: Bandit, Semgrep static analysis
- **Container Security Scan**: Trivy container vulnerability scanning
- **Secrets Detection**: TruffleHog, GitLeaks secret scanning
- **License Compliance**: License compatibility checking
- **Security Summary**: Automated security report generation

### 4. Release Pipeline (`release.yml`)
**Triggers**: Version tags (v*), Manual dispatch
**Purpose**: Automated package building, testing, and publishing

**Jobs:**
- **Pre-release Validation**: Version consistency, full test suite, security checks
- **Build Package**: Python package building and validation
- **Generate Release Notes**: Automated changelog generation
- **Publish to PyPI**: Automated PyPI package publishing
- **Create GitHub Release**: GitHub release creation with assets
- **Post-release Validation**: Installation testing and smoke tests
- **Notify Release**: Release success/failure notifications

## Configuration

### Environment Variables
- `PYTHON_VERSION`: Primary Python version (default: '3.9')
- `NODE_VERSION`: Node.js version for frontend (default: '18')

### Secrets Required
- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `GITHUB_TOKEN`: Automatically provided by GitHub

### Environments
- `pypi`: Protected environment for PyPI publishing (requires approval)

## Usage

### Running Workflows Manually

1. **Benchmarking**: Go to Actions → Comprehensive Benchmarking → Run workflow
   - Select benchmark type: all, perception, memory, performance, regression

2. **Security Scanning**: Go to Actions → Security Scanning → Run workflow

3. **Release**: Go to Actions → Release Pipeline → Run workflow
   - Provide version tag (e.g., v1.0.0)

### Monitoring Workflows

- **Dashboard**: View all workflow runs in the Actions tab
- **Notifications**: Receive email notifications for failures
- **Artifacts**: Download test results, benchmark data, security reports
- **Logs**: Detailed logs for debugging failed workflows

## Benchmark Results

### Performance Thresholds
- **Regression Threshold**: 15% performance degradation triggers failure
- **Warning Threshold**: 5% performance degradation shows warning
- **Memory Limit**: 1GB memory usage limit
- **Execution Time**: 5-minute maximum execution time

### Benchmark Types
1. **Perception Metrics**: Agent observation system performance
2. **Memory Database**: Database operation performance
3. **Redis Memory**: Redis-based memory system performance
4. **Observation Flow**: Observation generation throughput
5. **Pragma Profile**: Database configuration performance

## Security Scanning

### Vulnerability Severity Levels
- **High**: Critical security issues (failures)
- **Medium**: Important security issues (warnings)
- **Low**: Minor security issues (info)

### Scanned Components
- **Dependencies**: Known vulnerability database checks
- **Static Code**: Security anti-pattern detection
- **Secrets**: API keys, passwords, tokens
- **Licenses**: License compatibility validation

## Best Practices

### For Developers
1. **Run Tests Locally**: Use `pytest tests/` before pushing
2. **Check Code Quality**: Run `black`, `ruff`, `pylint` locally
3. **Security Review**: Address security warnings promptly
4. **Performance Awareness**: Monitor benchmark results for regressions

### For Maintainers
1. **Review PR Comments**: Automated benchmark and security summaries
2. **Monitor Workflow Health**: Check for consistent failures
3. **Update Dependencies**: Regular security and performance updates
4. **Release Management**: Use semantic versioning for tags

## Troubleshooting

### Common Issues

1. **Test Failures**
   - Check test logs for specific failures
   - Ensure all dependencies are properly installed
   - Verify test data and fixtures

2. **Benchmark Failures**
   - Check for performance regressions
   - Verify benchmark configuration
   - Review system resource usage

3. **Security Warnings**
   - Review security reports in artifacts
   - Update vulnerable dependencies
   - Address code security issues

4. **Release Failures**
   - Verify version consistency
   - Check PyPI credentials
   - Review release notes generation

### Getting Help

- **Workflow Logs**: Detailed execution logs in Actions tab
- **Artifacts**: Download reports and results for analysis
- **Issues**: Create GitHub issues for workflow problems
- **Documentation**: Refer to individual workflow files for details

## Customization

### Adding New Benchmarks
1. Create benchmark in `benchmarks/implementations/`
2. Add to `benchmark.yml` workflow
3. Update configuration in `config.yml`

### Adding New Security Checks
1. Install security tool in workflow
2. Add scanning step
3. Configure reporting and thresholds

### Modifying Test Matrix
1. Update `python_versions` in `config.yml`
2. Adjust test configurations
3. Update coverage thresholds

## Performance Optimization

### Workflow Optimization
- **Parallel Jobs**: Run independent jobs in parallel
- **Caching**: Cache dependencies and build artifacts
- **Matrix Strategy**: Test multiple Python versions efficiently
- **Conditional Execution**: Skip unnecessary jobs based on changes

### Resource Management
- **Memory Limits**: Monitor and limit memory usage
- **Timeout Settings**: Prevent hanging workflows
- **Artifact Cleanup**: Remove old artifacts to save space
- **Concurrent Limits**: Control concurrent workflow runs