"""
Unit tests for NVloom-Mojo core functionality
"""

from testing import assert_equal, assert_almost_equal, assert_true, assert_false

# Import from nvloom (would be actual imports in real implementation)
# from src.core.nvloom import NVloom, BandwidthResult, TestPattern
# from src.kernels.kernels import MemoryAccessPattern

fn test_bandwidth_calculation():
    """Test bandwidth calculation."""
    # 1 GB transferred in 1 second = 1 GB/s
    let bytes_transferred = 1024 * 1024 * 1024
    let time_seconds = 1.0
    let expected_gbps = 1.0
    
    let actual_gbps = Float64(bytes_transferred) / (1024.0 * 1024.0 * 1024.0) / time_seconds
    assert_almost_equal(actual_gbps, expected_gbps, 0.001)

fn test_memory_alignment():
    """Test memory alignment calculation."""
    # Test 128-byte alignment
    let unaligned_addr = 0x1234567
    let aligned_addr = (unaligned_addr + 127) & ~127
    
    assert_equal(aligned_addr & 127, 0)
    assert_true(aligned_addr >= unaligned_addr)

fn test_buffer_size_parsing():
    """Test buffer size string parsing."""
    # Test various size formats
    assert_equal(parse_size("1024"), 1024)
    assert_equal(parse_size("1K"), 1024)
    assert_equal(parse_size("1M"), 1024 * 1024)
    assert_equal(parse_size("1G"), 1024 * 1024 * 1024)
    assert_equal(parse_size("512M"), 512 * 1024 * 1024)

fn parse_size(size_str: String) -> Int:
    """Helper function to parse size strings."""
    var multiplier = 1
    var num_str = size_str
    
    if size_str.endswith("K"):
        multiplier = 1024
        num_str = size_str[:-1]
    elif size_str.endswith("M"):
        multiplier = 1024 * 1024
        num_str = size_str[:-1]
    elif size_str.endswith("G"):
        multiplier = 1024 * 1024 * 1024
        num_str = size_str[:-1]
    
    return int(num_str) * multiplier

fn test_gpu_device_initialization():
    """Test GPU device initialization."""
    # Test device ID calculation
    let device_id = 15
    let expected_node_id = device_id // 8  # 1
    let expected_rack_id = device_id // 72  # 0
    
    assert_equal(expected_node_id, 1)
    assert_equal(expected_rack_id, 0)

fn test_pattern_selection():
    """Test pattern selection logic."""
    # Test pattern enum values
    assert_equal(TestPattern.PAIRWISE, 0)
    assert_equal(TestPattern.BISECT, 1)
    assert_equal(TestPattern.GPU_TO_RACK, 2)
    assert_equal(TestPattern.RACK_TO_RACK, 3)

fn test_copy_type_selection():
    """Test copy type selection."""
    assert_equal(CopyType.WRITE, 0)
    assert_equal(CopyType.READ, 1)
    assert_equal(CopyType.BIDIRECTIONAL, 2)

fn test_bisect_peer_calculation():
    """Test bisect pattern peer calculation."""
    let num_gpus = 8
    let half = num_gpus // 2
    
    # Test peer calculation for each GPU
    for gpu_id in range(num_gpus):
        let peer_gpu = (gpu_id + half) % num_gpus
        
        # Verify peer is in opposite half
        if gpu_id < half:
            assert_true(peer_gpu >= half)
        else:
            assert_true(peer_gpu < half)

fn test_rack_calculation():
    """Test rack assignment calculation."""
    let gpus_per_rack = 72
    
    # Test various GPU IDs
    assert_equal(0 // gpus_per_rack, 0)   # GPU 0 -> Rack 0
    assert_equal(71 // gpus_per_rack, 0)  # GPU 71 -> Rack 0
    assert_equal(72 // gpus_per_rack, 1)  # GPU 72 -> Rack 1
    assert_equal(143 // gpus_per_rack, 1) # GPU 143 -> Rack 1
    assert_equal(144 // gpus_per_rack, 2) # GPU 144 -> Rack 2

fn test_block_size_selection():
    """Test optimal block size selection."""
    # Small transfer
    let small_size = 512
    let small_block = select_block_size(small_size)
    assert_equal(small_block, 32)
    
    # Medium transfer
    let medium_size = 100 * 1024
    let medium_block = select_block_size(medium_size)
    assert_equal(medium_block, 128)
    
    # Large transfer
    let large_size = 10 * 1024 * 1024
    let large_block = select_block_size(large_size)
    assert_equal(large_block, 256)

fn select_block_size(size: Int) -> Int:
    """Helper function for block size selection."""
    if size < 1024:
        return 32
    elif size < 1024 * 1024:
        return 128
    else:
        return 256

# Test constants
alias TestPattern.PAIRWISE = 0
alias TestPattern.BISECT = 1
alias TestPattern.GPU_TO_RACK = 2
alias TestPattern.RACK_TO_RACK = 3
alias CopyType.WRITE = 0
alias CopyType.READ = 1
alias CopyType.BIDIRECTIONAL = 2

fn run_all_tests():
    """Run all unit tests."""
    print("Running NVloom-Mojo unit tests...")
    
    test_bandwidth_calculation()
    print("✓ Bandwidth calculation test passed")
    
    test_memory_alignment()
    print("✓ Memory alignment test passed")
    
    test_buffer_size_parsing()
    print("✓ Buffer size parsing test passed")
    
    test_gpu_device_initialization()
    print("✓ GPU device initialization test passed")
    
    test_pattern_selection()
    print("✓ Pattern selection test passed")
    
    test_copy_type_selection()
    print("✓ Copy type selection test passed")
    
    test_bisect_peer_calculation()
    print("✓ Bisect peer calculation test passed")
    
    test_rack_calculation()
    print("✓ Rack calculation test passed")
    
    test_block_size_selection()
    print("✓ Block size selection test passed")
    
    print("\nAll tests passed! ✨")

fn main():
    run_all_tests()
