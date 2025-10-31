"""
Example usage of NVloom Mojo library
Demonstrates various test patterns and configurations
"""

from nvloom import NVloom, TestPattern, CopyType, CopyEngine, BandwidthResult
from kernels import AsyncCopyManager, MemoryAccessPattern, PerformanceCounters

fn example_basic_usage():
    """Basic usage example"""
    print("=== Basic NVloom Usage ===")
    
    # Initialize with 4 GPUs
    var nvloom = NVloom(4)
    
    # Run a simple bisect test
    let result = nvloom.run_testcase(
        TestPattern.BISECT,
        CopyType.WRITE,
        CopyEngine.SM
    )
    
    print("Test completed:", result.test_name)
    print("Mean bandwidth:", result.summary_stats.mean_bandwidth, "GB/s")
    
    # Cleanup
    nvloom.cleanup()

fn example_custom_configuration():
    """Example with custom configuration"""
    print("\n=== Custom Configuration ===")
    
    var nvloom = NVloom(8)
    
    # Configure parameters
    nvloom.buffer_size = 1024 * 1024 * 1024  # 1GB
    nvloom.iterations = 32
    
    # Run pairwise test with custom settings
    let result = nvloom.run_testcase(
        TestPattern.PAIRWISE,
        CopyType.BIDIRECTIONAL,
        CopyEngine.CE
    )
    
    # Print detailed results
    print("Pairwise test results:")
    print("  Total measurements:", len(result.measurements))
    print("  Min bandwidth:", result.summary_stats.min_bandwidth, "GB/s")
    print("  Max bandwidth:", result.summary_stats.max_bandwidth, "GB/s")
    print("  Mean bandwidth:", result.summary_stats.mean_bandwidth, "GB/s")
    
    nvloom.cleanup()

fn example_suite_execution():
    """Run a complete test suite"""
    print("\n=== Test Suite Execution ===")
    
    var nvloom = NVloom(16)
    
    # Run fabric stress suite
    let results = nvloom.run_suite("fabric-stress")
    
    print("Suite completed with", len(results), "tests")
    
    for i in range(len(results)):
        let test = results[i]
        print(f"  Test {i+1}: {test.test_name}")
        print(f"    Mean: {test.summary_stats.mean_bandwidth:.2f} GB/s")
    
    nvloom.cleanup()

fn example_async_operations():
    """Example using async copy operations"""
    print("\n=== Async Copy Operations ===")
    
    # Create async copy manager
    var async_mgr = AsyncCopyManager(num_streams=4)
    
    # Allocate buffers
    let size = 256 * 1024 * 1024  # 256MB
    let src = UnsafePointer[UInt8].alloc(size)
    let dst = UnsafePointer[UInt8].alloc(size)
    
    # Launch async copies on different streams
    for stream in range(4):
        let stream_id = async_mgr.async_copy(src, dst, size // 4, stream)
        print(f"Launched copy on stream {stream_id}")
    
    # Wait for completion
    async_mgr.synchronize_all()
    print("All async copies completed")
    
    # Cleanup
    src.free()
    dst.free()

fn example_performance_monitoring():
    """Monitor performance metrics"""
    print("\n=== Performance Monitoring ===")
    
    var perf_counters = PerformanceCounters()
    var nvloom = NVloom(8)
    
    # Run multiple tests and collect metrics
    for i in range(3):
        let result = nvloom.run_testcase(
            TestPattern.BISECT,
            CopyType.WRITE,
            CopyEngine.SM
        )
        
        # Record performance data
        for j in range(len(result.measurements)):
            let m = result.measurements[j]
            let bytes_transferred = nvloom.buffer_size
            let time_ns = Int(Float64(bytes_transferred) / (m.bandwidth_gbps * 1.0))
            perf_counters.record_transfer(bytes_transferred, time_ns)
    
    # Print performance statistics
    perf_counters.print_stats()
    
    nvloom.cleanup()

fn example_memory_pattern_analysis():
    """Analyze and optimize memory access patterns"""
    print("\n=== Memory Pattern Analysis ===")
    
    # Allocate test buffers
    let size = 512 * 1024 * 1024
    let aligned_src = UnsafePointer[UInt8].alloc(size + 128)
    let aligned_dst = UnsafePointer[UInt8].alloc(size + 128)
    
    # Ensure 128-byte alignment
    let src = (aligned_src.address + 127) & ~127
    let dst = (aligned_dst.address + 127) & ~127
    
    # Analyze access pattern
    let pattern = MemoryAccessPattern.analyze_pattern(
        UnsafePointer[UInt8](src),
        UnsafePointer[UInt8](dst),
        size
    )
    
    if pattern == 0:
        print("Using vectorized copy (optimal alignment)")
    elif pattern == 1:
        print("Using coalesced copy")
    else:
        print("Using simple copy")
    
    # Select optimal configuration
    let block_size = MemoryAccessPattern.select_block_size(size)
    let grid_size = MemoryAccessPattern.select_grid_size(size, block_size)
    
    print(f"Optimal configuration: {grid_size} blocks x {block_size} threads")
    
    # Cleanup
    aligned_src.free()
    aligned_dst.free()

fn example_custom_test_pattern():
    """Create and run a custom test pattern"""
    print("\n=== Custom Test Pattern ===")
    
    var nvloom = NVloom(8)
    
    # Create custom ring pattern (each GPU copies to next)
    var custom_result = TestResult("ring_pattern", -1)
    
    for gpu in range(nvloom.num_gpus):
        let next_gpu = (gpu + 1) % nvloom.num_gpus
        
        # Allocate buffers
        let src_buf = UnsafePointer[UInt8].alloc(nvloom.buffer_size)
        let dst_buf = UnsafePointer[UInt8].alloc(nvloom.buffer_size)
        
        # Measure bandwidth (simulated)
        let bandwidth = 100.0 + Float64(gpu) * 5.0  # Simulated values
        
        custom_result.add_measurement(
            BandwidthResult(gpu, next_gpu, bandwidth, 1.5)
        )
        
        # Cleanup
        src_buf.free()
        dst_buf.free()
    
    print("Ring pattern test completed")
    print("  Total hops:", nvloom.num_gpus)
    print("  Average bandwidth:", custom_result.summary_stats.mean_bandwidth, "GB/s")
    
    nvloom.cleanup()

fn example_multicast_operations():
    """Example of multicast operations"""
    print("\n=== Multicast Operations ===")
    
    var nvloom = NVloom(8)
    
    # Note: Multicast requires special hardware support
    print("Multicast test configuration:")
    print("  GPUs involved:", nvloom.num_gpus)
    print("  Multicast groups: All GPUs")
    print("  Operation: Broadcast write")
    
    # Run multicast test (if available)
    # In real usage, would check hardware capability first
    let result = nvloom.run_testcase(
        TestPattern.MULTICAST_ONE_TO_ALL,
        CopyType.WRITE,
        CopyEngine.SM
    )
    
    print("Multicast bandwidth:", result.summary_stats.mean_bandwidth, "GB/s")
    print("Effective bandwidth:", 
          result.summary_stats.mean_bandwidth * (nvloom.num_gpus - 1), "GB/s")
    
    nvloom.cleanup()

fn main():
    """Run all examples"""
    print("NVloom Mojo Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_configuration()
    example_suite_execution()
    example_async_operations()
    example_performance_monitoring()
    example_memory_pattern_analysis()
    example_custom_test_pattern()
    example_multicast_operations()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
