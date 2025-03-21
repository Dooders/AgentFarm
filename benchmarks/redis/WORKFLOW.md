# Redis Benchmark GitHub Action

This GitHub Action automatically runs the Redis memory benchmark suite whenever changes are made to the Redis memory implementation code, providing immediate feedback on performance impacts.

## How It Works

1. **Trigger**: The workflow runs automatically when:
   - Changes are pushed to the `main` branch that affect Redis memory code
   - Changes are made to files in the `benchmarks/redis/**` directory
   - The workflow file itself is changed
   - Manually triggered via GitHub Actions interface

2. **Environment**: The workflow:
   - Runs on Ubuntu latest
   - Sets up a Redis server using Docker
   - Installs Python 3.9 and required dependencies

3. **Benchmark Process**:
   - Runs the simple Redis benchmark with different batch sizes (10, 100, 500)
   - Generates visualization charts
   - Tracks performance changes over time
   - Archives all results as workflow artifacts
   - Updates the `benchmarks/redis/results/` directory in the repo

## Viewing Results

### During Workflow Run

You can watch the benchmark as it runs in the GitHub Actions tab of your repository.

### After Completion

1. **In the Repository**:
   - Check the `benchmarks/redis/results/` directory for the latest results
   - Review the `benchmarks/redis/results/README.md` for a summary
   - View the charts in `benchmarks/redis/results/charts/`

2. **As Workflow Artifacts**:
   - Go to the completed workflow run
   - Download the "benchmark-results" artifact
   - Unzip to view JSON data files and charts

## Performance Tracking

The workflow maintains a history of benchmark results over time, allowing you to:
- Track performance changes between versions
- Identify regressions
- Verify performance improvements

## Manually Triggering

To manually run the benchmark:
1. Go to the "Actions" tab in your repository
2. Select the "Redis Memory Benchmark" workflow
3. Click "Run workflow"
4. Choose the branch to run against
5. Click "Run workflow"

## Extending the Benchmark

To add more benchmarks or metrics:

1. Modify `benchmarks/redis/simple_redis_benchmark.py` to add new tests
2. Update `benchmarks/redis/advanced_charts.py` to create new visualizations
3. Edit `.github/workflows/redis_benchmark.yml` to include the new tests

## Notifications

By default, you'll receive GitHub notifications for:
- Workflow failures
- New workflow runs (if watching the repository)

To set up additional notifications (e.g., Slack, email), add notification steps to the workflow file. 