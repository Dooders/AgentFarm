# AgentMemory Testing Strategy

This document outlines the comprehensive testing approach for the AgentMemory system, ensuring reliability, performance, and correctness across all components.

## 1. Unit Testing

### 1.1 Memory Agent Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_memory_initialization` | Verify correct initialization of memory tiers | All stores properly initialized with correct agent ID |
| `test_store_agent_state` | Test storing a new agent state | State correctly stored in STM tier |
| `test_retrieve_recent_state` | Test retrieving recently stored state | Retrieved state matches stored state |
| `test_memory_compression` | Test compression functionality | Compressed state retains essential information |
| `test_state_transition` | Test state transition between memory tiers | State correctly moves from STM → IM → LTM |
| `test_importance_calculation` | Test importance score calculation | Scores reflect state significance |
| `test_memory_hooks` | Test memory event hooks | Hooks trigger at appropriate times |
| `test_memory_agent_config` | Test configuration parameters | Agent behavior reflects configuration |

### 1.2 Redis Integration Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_redis_connection` | Test Redis connection establishment | Connection successful with proper error handling |
| `test_redis_state_storage` | Test state hash structure in Redis | Hash fields match schema definition |
| `test_redis_timeline_operations` | Test timeline sorted set operations | Entries ordered correctly by step number |
| `test_redis_index_consistency` | Test index structure maintenance | Indices updated when states added/modified |
| `test_redis_ttl_expiration` | Test TTL-based expiration of IM tier | Entries expire after configured duration |
| `test_redis_batch_operations` | Test batch get/set operations | Batch operations work correctly |
| `test_redis_error_handling` | Test handling of Redis connectivity issues | Graceful degradation when Redis unavailable |

### 1.3 SQLite Integration Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_sqlite_connection` | Test SQLite connection establishment | Connection successful |
| `test_sqlite_state_storage` | Test state storage in SQLite | Entries match schema definition |
| `test_sqlite_query_performance` | Test query performance | Queries complete within time threshold |
| `test_sqlite_batch_insertion` | Test batch insertion from IM tier | Batch operations completed successfully |
| `test_sqlite_indexing` | Test index usage in queries | Queries use appropriate indices |
| `test_sqlite_error_handling` | Test DB error handling | Proper error handling and recovery |

### 1.4 Vector Embedding Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_autoencoder_initialization` | Test autoencoder model loading | Model loads successfully |
| `test_state_embedding_generation` | Test embedding vector generation | Embeddings have correct dimensions |
| `test_embedding_similarity` | Test similarity calculations | Similar states have higher similarity scores |
| `test_embedding_compression` | Test embedding dimension reduction | Compression preserves similarity relationships |
| `test_embedding_reconstruction` | Test state reconstruction from embeddings | Reconstructed state approximates original |

### 1.5 API Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_api_store_methods` | Test store_agent_state API | API correctly stores state |
| `test_api_retrieval_methods` | Test retrieve_states API | API correctly retrieves states |
| `test_api_parameter_validation` | Test API parameter validation | Invalid parameters rejected with clear errors |
| `test_api_error_handling` | Test API error handling | Errors handled gracefully with useful messages |
| `test_api_rate_limiting` | Test API rate limiting | Rate limits enforced correctly |

## 2. Integration Testing

### 2.1 Component Integration Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_stm_im_transition` | Test transition from STM to IM | States correctly compressed and moved |
| `test_im_ltm_transition` | Test transition from IM to LTM | States correctly compressed and persisted |
| `test_retrieval_across_tiers` | Test state retrieval across memory tiers | Correct state retrieved regardless of tier location |
| `test_memory_backfilling` | Test retrieval when STM misses but IM/LTM has the state | Correct backfilling behavior |
| `test_full_memory_lifecycle` | Test complete lifecycle of a memory entry | State correctly transitions through all tiers |

### 2.2 System Integration Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_agent_decision_integration` | Test integration with agent decision-making | Memory influences agent decisions correctly |
| `test_multiple_agent_isolation` | Test memory isolation between agents | No cross-contamination of agent memories |
| `test_simulation_integration` | Test integration with full simulation cycle | Memory system works within simulation |
| `test_recovery_from_failure` | Test system recovery after component failure | System recovers with minimal data loss |
| `test_concurrent_operations` | Test concurrent read/write operations | No corruption or race conditions |

### 2.3 End-to-End Tests

| Test Case | Description | Assertions |
|-----------|-------------|------------|
| `test_e2e_agent_memory_formation` | Test memory formation during agent operation | Memories formed and retrievable |
| `test_e2e_memory_based_decisions` | Test agent using past memories for decisions | Decisions influenced by past experiences |
| `test_e2e_long_running_simulation` | Test memory system in long-running simulation | System remains stable over time |
| `test_e2e_memory_persistence` | Test persistence across simulation restarts | Memory retrievable after restart |

## 3. Performance Benchmarking

### 3.1 Latency Benchmarks

| Benchmark | Description | Target Metric |
|-----------|-------------|---------------|
| `bench_stm_write_latency` | Measure STM write operation latency | < 5ms per operation |
| `bench_stm_read_latency` | Measure STM read operation latency | < 2ms per operation |
| `bench_im_write_latency` | Measure IM write operation latency | < 10ms per operation |
| `bench_im_read_latency` | Measure IM read operation latency | < 5ms per operation |
| `bench_ltm_write_latency` | Measure LTM write operation latency | < 20ms per operation |
| `bench_ltm_read_latency` | Measure LTM read operation latency | < 10ms per operation |
| `bench_vector_similarity_latency` | Measure vector similarity calculation latency | < 15ms for 100 comparisons |
| `bench_embedding_generation_latency` | Measure embedding generation latency | < 50ms per state |

### 3.2 Throughput Benchmarks

| Benchmark | Description | Target Metric |
|-----------|-------------|---------------|
| `bench_stm_write_throughput` | Measure STM write operations per second | > 1000 ops/sec |
| `bench_stm_read_throughput` | Measure STM read operations per second | > 2000 ops/sec |
| `bench_im_write_throughput` | Measure IM write operations per second | > 500 ops/sec |
| `bench_im_read_throughput` | Measure IM read operations per second | > 1000 ops/sec |
| `bench_ltm_write_throughput` | Measure LTM write operations per second | > 200 ops/sec |
| `bench_ltm_read_throughput` | Measure LTM read operations per second | > 500 ops/sec |
| `bench_batch_operations_throughput` | Measure batch operation throughput | > 5000 states/sec |

### 3.3 Memory Usage Benchmarks

| Benchmark | Description | Target Metric |
|-----------|-------------|---------------|
| `bench_memory_per_state` | Measure memory used per state in STM | < 2KB average |
| `bench_im_compression_ratio` | Measure IM compression efficiency | > 5:1 ratio |
| `bench_ltm_compression_ratio` | Measure LTM compression efficiency | > 20:1 ratio |
| `bench_scaling_memory_usage` | Measure memory scaling with state count | Sub-linear scaling |
| `bench_redis_memory_efficiency` | Measure Redis memory utilization | < 75% overhead |

### 3.4 Scalability Benchmarks

| Benchmark | Description | Target Metric |
|-----------|-------------|---------------|
| `bench_multi_agent_scaling` | Measure performance scaling with agent count | Linear degradation |
| `bench_simulation_size_scaling` | Measure scaling with simulation size | Linear degradation |
| `bench_timeline_length_scaling` | Measure performance vs. history length | Logarithmic degradation |
| `bench_concurrent_access_scaling` | Measure scaling with concurrent operations | Graceful degradation |

## 4. Testing Tools and Infrastructure

### 4.1 Unit Test Framework
- **PyTest**: Primary framework for unit and integration tests
- **PyTest-Redis**: For Redis integration testing with ephemeral instances
- **PyTest-SQLite**: For SQLite testing with in-memory databases

### 4.2 Performance Testing Tools
- **Locust**: For load testing API endpoints
- **PyTest-Benchmark**: For fine-grained performance measurements
- **Redis-Benchmark**: For Redis-specific performance testing
- **Memory-Profiler**: For memory usage tracking

### 4.3 CI/CD Integration
- Automated test execution on PR creation
- Performance benchmark tracking across commits
- Regression detection for performance metrics
- Code coverage reporting

## 5. Testing Environments

### 5.1 Local Development Environment
- Docker Compose setup with Redis
- Isolated environment for repeatable tests
- Mock data generation for consistent testing

### 5.2 CI Environment
- Containerized test execution
- Resource-constrained environment to catch performance issues
- Matrix testing across different Redis versions

### 5.3 Staging Environment
- Production-like configuration
- Scaled-down but representative data volumes
- Integration with other system components

## 6. Quality Metrics and Targets

| Metric | Target |
|--------|--------|
| Unit Test Coverage | > 90% |
| Integration Test Coverage | > 80% |
| Maximum Acceptable Latency | < 100ms for any operation |
| Throughput Minimum | > 100 operations/sec under load |
| Memory Efficiency | < 1GB for 100k agent states |
| Error Rate | < 0.01% under normal operation |
| Recovery Time | < 5 seconds after component failure |

## 7. Test Data Management

### 7.1 Test Data Generation
- Synthetic agent state generation
- Realistic simulation patterns
- Edge case state construction

### 7.2 Test Data Versioning
- Version-controlled test datasets
- Regression testing with historical data
- Reproducible test scenarios

## 8. Testing Schedule

| Phase | Timing | Focus |
|-------|--------|-------|
| Pre-implementation | Before development | API contract testing, mock interfaces |
| Component Development | During development | Unit tests for each component |
| Integration | After components complete | Cross-component integration |
| System Testing | Before release | End-to-end verification |
| Performance Testing | Before and after release | Benchmark validation |
| Continuous Testing | Ongoing | Regression detection |

## 9. Acceptance Criteria

The testing is considered successful when:

1. All unit tests pass with > 90% coverage
2. All integration tests pass with > 80% coverage
3. Performance benchmarks meet or exceed target metrics
4. No critical or high-severity bugs remain open
5. System demonstrates stability under load
6. Memory utilization remains within acceptable bounds
7. All error conditions are handled gracefully 