@echo off
REM Redis Benchmark Runner Script for Windows
REM This script runs Redis benchmarks and generates visualizations

SETLOCAL EnableDelayedExpansion

REM Default settings
SET ENTRIES=500
SET NOTE=
SET QUICK=false
SET COMPARE=false
SET BATCH_SIZES=10 100 500
SET USE_DOCKER=false
SET START_REDIS=false

REM Parse command line arguments
:parse_args
IF "%~1"=="" GOTO end_parse_args
IF "%~1"=="-e" (
    SET ENTRIES=%~2
    SHIFT
    GOTO next_arg
)
IF "%~1"=="--entries" (
    SET ENTRIES=%~2
    SHIFT
    GOTO next_arg
)
IF "%~1"=="-n" (
    SET NOTE=%~2
    SHIFT
    GOTO next_arg
)
IF "%~1"=="--note" (
    SET NOTE=%~2
    SHIFT
    GOTO next_arg
)
IF "%~1"=="-q" (
    SET QUICK=true
    GOTO next_arg
)
IF "%~1"=="--quick" (
    SET QUICK=true
    GOTO next_arg
)
IF "%~1"=="-c" (
    SET COMPARE=true
    GOTO next_arg
)
IF "%~1"=="--compare" (
    SET COMPARE=true
    GOTO next_arg
)
IF "%~1"=="-b" (
    SET BATCH_SIZES=%~2
    SHIFT
    GOTO next_arg
)
IF "%~1"=="--batch-sizes" (
    SET BATCH_SIZES=%~2
    SHIFT
    GOTO next_arg
)
IF "%~1"=="--docker" (
    SET USE_DOCKER=true
    GOTO next_arg
)
IF "%~1"=="--start-redis" (
    SET START_REDIS=true
    GOTO next_arg
)
IF "%~1"=="-h" (
    CALL :print_usage
    exit /b 0
)
IF "%~1"=="--help" (
    CALL :print_usage
    exit /b 0
)

echo Unknown option: %~1
CALL :print_usage
exit /b 1

:next_arg
SHIFT
GOTO parse_args

:end_parse_args

REM If using Docker, check Docker is available
IF "%USE_DOCKER%"=="true" (
    docker -v > nul 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo ⚠️  Docker doesn't appear to be installed or running
        echo Please install Docker and try again
        exit /b 1
    )
    
    IF "%START_REDIS%"=="true" (
        echo 🐳 Starting Redis using Docker...
        docker-compose -f ..\..\docker-compose.yml up -d
        
        REM Wait a moment for Redis to start
        timeout /t 3 /nobreak > nul
    )
)

REM Ensure Redis is running
redis-cli ping > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    IF "%USE_DOCKER%"=="true" (
        REM Try with Docker exec if using Docker
        docker exec -i redis-benchmark redis-cli ping > nul 2>&1
        IF %ERRORLEVEL% NEQ 0 (
            echo ⚠️  Redis server doesn't appear to be running in Docker
            echo Please start Redis with: docker-compose -f ..\..\docker-compose.yml up -d
            exit /b 1
        )
    ) ELSE (
        echo ⚠️  Redis server doesn't appear to be running
        echo Please start Redis server before running benchmarks
        echo or use --docker --start-redis to use Docker
        exit /b 1
    )
)

REM Ensure output directories exist
if not exist benchmark_results mkdir benchmark_results
if not exist charts mkdir charts

REM If quick mode, reduce entries
IF "%QUICK%"=="true" (
    SET ENTRIES=100
    echo 🚀 Running quick benchmark with !ENTRIES! entries
)

REM Run single benchmark or comparison
IF "%COMPARE%"=="true" (
    echo 📊 Running comparison benchmark with batch sizes: %BATCH_SIZES%
    
    REM Clear any previous results
    DEL /Q benchmark_results\batch_*.json 2>nul
    
    REM Run benchmarks with different batch sizes
    FOR %%S IN (%BATCH_SIZES%) DO (
        echo Running benchmark with batch size %%S...
        IF "%USE_DOCKER%"=="true" (
            python simple_redis_benchmark.py --redis-env DOCKER --memory-entries %ENTRIES% --batch-size %%S --output benchmark_results/batch_%%S.json
        ) ELSE (
            python simple_redis_benchmark.py --memory-entries %ENTRIES% --batch-size %%S --output benchmark_results/batch_%%S.json
        )
    )
    
    REM Generate charts
    echo Generating comparison charts...
    SET FILES=
    FOR %%S IN (%BATCH_SIZES%) DO (
        SET FILES=!FILES! benchmark_results/batch_%%S.json
    )
    python advanced_charts.py %FILES%
    
    REM Add to history if note is provided
    IF NOT "%NOTE%"=="" (
        echo Adding results to performance history...
        REM For Windows, we'll use the first batch size as reference
        FOR /F "tokens=1" %%S IN ("%BATCH_SIZES%") DO (
            python track_performance.py add benchmark_results/batch_%%S.json --note "%NOTE%"
        )
    )
    
    REM Show performance history
    python track_performance.py show
    
    REM Generate performance chart
    python track_performance.py plot
    
    echo ✅ Benchmark complete! Results saved to benchmark_results/ and charts/
    
) ELSE (
    REM Run single benchmark
    echo 📊 Running single benchmark with %ENTRIES% entries...
    
    REM Default batch size
    SET BATCH_SIZE=100
    
    REM Run benchmark
    IF "%USE_DOCKER%"=="true" (
        python simple_redis_benchmark.py --redis-env DOCKER --memory-entries %ENTRIES% --batch-size %BATCH_SIZE% --output benchmark_results/latest.json
    ) ELSE (
        python simple_redis_benchmark.py --memory-entries %ENTRIES% --batch-size %BATCH_SIZE% --output benchmark_results/latest.json
    )
    
    REM Add to history if note is provided
    IF NOT "%NOTE%"=="" (
        echo Adding results to performance history...
        python track_performance.py add benchmark_results/latest.json --note "%NOTE%"
        
        REM Generate performance chart
        python track_performance.py plot
    )
    
    echo ✅ Benchmark complete! Results saved to benchmark_results/latest.json
)

echo.
echo 📈 To view performance history: python track_performance.py show
echo 📊 To generate more charts: python advanced_charts.py benchmark_results/*.json

REM Stop Docker Redis if we started it
IF "%USE_DOCKER%"=="true" IF "%START_REDIS%"=="true" (
    echo 🐳 Do you want to stop the Redis Docker container? [Y/N]
    set /p stop_docker=
    if /i "!stop_docker!"=="y" (
        echo Stopping Redis Docker container...
        docker-compose -f ..\..\docker-compose.yml down
    )
)

GOTO :eof

:print_usage
echo Usage: %0 [options]
echo.
echo Options:
echo   -e, --entries N       Number of memory entries (default: 500)
echo   -n, --note TEXT       Add a note to the benchmark
echo   -q, --quick           Run a quick benchmark (fewer entries)
echo   -c, --compare         Run comparison with different batch sizes
echo   -b, --batch-sizes     Specify batch sizes to compare (default: "10 100 500")
echo   --docker              Use Redis from Docker container
echo   --start-redis         Start Redis Docker container before benchmark
echo   -h, --help            Display this help message
echo.
echo Examples:
echo   %0 --quick                   # Run a quick benchmark
echo   %0 --compare                 # Compare different batch sizes
echo   %0 --entries 1000 --note "New JSON serializer test"
echo   %0 --docker --start-redis    # Use Redis from Docker
GOTO :eof 