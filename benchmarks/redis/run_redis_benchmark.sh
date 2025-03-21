#!/bin/bash
# Redis Benchmark Runner Script
# This script runs Redis benchmarks and generates visualizations

# Parse command line arguments
ENTRIES=500
NOTE=""
QUICK=false
COMPARE=false
BATCH_SIZES="10 100 500"
USE_DOCKER=false
START_REDIS=false

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -e, --entries N       Number of memory entries (default: 500)"
    echo "  -n, --note TEXT       Add a note to the benchmark"
    echo "  -q, --quick           Run a quick benchmark (fewer entries)"
    echo "  -c, --compare         Run comparison with different batch sizes"
    echo "  -b, --batch-sizes     Specify batch sizes to compare (default: \"10 100 500\")"
    echo "  --docker              Use Redis from Docker container"
    echo "  --start-redis         Start Redis Docker container before benchmark"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --quick                   # Run a quick benchmark"
    echo "  $0 --compare                 # Compare different batch sizes"
    echo "  $0 --entries 1000 --note \"New JSON serializer test\""
    echo "  $0 --docker --start-redis    # Use Redis from Docker"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--entries)
            ENTRIES="$2"
            shift 2
            ;;
        -n|--note)
            NOTE="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        -c|--compare)
            COMPARE=true
            shift
            ;;
        -b|--batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --start-redis)
            START_REDIS=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# If using Docker, check Docker is available
if [ "$USE_DOCKER" = true ]; then
    if ! command -v docker &> /dev/null; then
        echo "⚠️  Docker doesn't appear to be installed or in PATH"
        echo "Please install Docker and try again"
        exit 1
    fi
    
    if [ "$START_REDIS" = true ]; then
        echo "🐳 Starting Redis using Docker..."
        docker-compose up -d
        
        # Wait a moment for Redis to start
        sleep 3
    fi
fi

# Ensure Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    if [ "$USE_DOCKER" = true ]; then
        # Try with Docker exec if using Docker
        if ! docker exec -i redis-benchmark redis-cli ping > /dev/null 2>&1; then
            echo "⚠️  Redis server doesn't appear to be running in Docker"
            echo "Please start Redis with: docker-compose up -d"
            exit 1
        fi
    else
        echo "⚠️  Redis server doesn't appear to be running"
        echo "Please start Redis server before running benchmarks"
        echo "or use --docker --start-redis to use Docker"
        exit 1
    fi
fi

# Ensure output directories exist
mkdir -p benchmark_results
mkdir -p charts

# If quick mode, reduce entries
if [ "$QUICK" = true ]; then
    ENTRIES=100
    echo "🚀 Running quick benchmark with $ENTRIES entries"
fi

# Run single benchmark or comparison
if [ "$COMPARE" = true ]; then
    echo "📊 Running comparison benchmark with batch sizes: $BATCH_SIZES"
    
    # Clear any previous results
    rm -f benchmark_results/batch_*.json
    
    # Run benchmarks with different batch sizes
    for SIZE in $BATCH_SIZES; do
        echo "Running benchmark with batch size $SIZE..."
        if [ "$USE_DOCKER" = true ]; then
            python simple_redis_benchmark.py --redis-env DOCKER --memory-entries $ENTRIES --batch-size $SIZE --output benchmark_results/batch_$SIZE.json
        else
            python simple_redis_benchmark.py --memory-entries $ENTRIES --batch-size $SIZE --output benchmark_results/batch_$SIZE.json
        fi
    done
    
    # Generate result files array for advanced charts
    FILES=""
    for SIZE in $BATCH_SIZES; do
        FILES="$FILES benchmark_results/batch_$SIZE.json"
    done
    
    # Generate charts
    echo "Generating comparison charts..."
    python advanced_charts.py $FILES
    
    # Add to history if note is provided
    if [ -n "$NOTE" ]; then
        echo "Adding results to performance history..."
        # Use the middle batch size as the reference
        MID_SIZE=$(echo $BATCH_SIZES | tr ' ' '\n' | sort -n | awk 'NR==2{print}')
        if [ -z "$MID_SIZE" ]; then
            # If only one batch size, use that
            MID_SIZE=$(echo $BATCH_SIZES | tr ' ' '\n' | sort -n | head -1)
        fi
        python track_performance.py add benchmark_results/batch_$MID_SIZE.json --note "$NOTE"
    fi
    
    # Show performance history
    python track_performance.py show
    
    # Generate performance chart
    python track_performance.py plot
    
    echo "✅ Benchmark complete! Results saved to benchmark_results/ and charts/"
    
else
    # Run single benchmark
    echo "📊 Running single benchmark with $ENTRIES entries..."
    
    # Default batch size
    BATCH_SIZE=100
    
    # Run benchmark
    if [ "$USE_DOCKER" = true ]; then
        python simple_redis_benchmark.py --redis-env DOCKER --memory-entries $ENTRIES --batch-size $BATCH_SIZE --output benchmark_results/latest.json
    else
        python simple_redis_benchmark.py --memory-entries $ENTRIES --batch-size $BATCH_SIZE --output benchmark_results/latest.json
    fi
    
    # Add to history if note is provided
    if [ -n "$NOTE" ]; then
        echo "Adding results to performance history..."
        python track_performance.py add benchmark_results/latest.json --note "$NOTE"
        
        # Generate performance chart
        python track_performance.py plot
    fi
    
    echo "✅ Benchmark complete! Results saved to benchmark_results/latest.json"
fi

echo ""
echo "📈 To view performance history: python track_performance.py show"
echo "📊 To generate more charts: python advanced_charts.py benchmark_results/*.json"

# Stop Docker Redis if we started it
if [ "$USE_DOCKER" = true ] && [ "$START_REDIS" = true ]; then
    echo "🐳 Do you want to stop the Redis Docker container? [y/N]"
    read -r stop_docker
    if [[ "$stop_docker" =~ ^[Yy]$ ]]; then
        echo "Stopping Redis Docker container..."
        docker-compose down
    fi
fi 