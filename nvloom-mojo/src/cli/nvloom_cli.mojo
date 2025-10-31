"""
NVloom CLI - Command Line Interface for NVloom Mojo
Provides MPI support for distributed GPU testing
"""

from python import Python
import sys
from collections import Dict, DynamicVector

# Import main NVloom library (would be actual import in real Mojo)
# from nvloom import NVloom, TestPattern, CopyType, CopyEngine

# ============================================================================
# MPI Wrapper
# ============================================================================

struct MPIWrapper:
    """Wrapper for MPI functionality"""
    var rank: Int
    var size: Int
    var comm: PythonObject
    
    fn __init__(inout self) raises:
        """Initialize MPI"""
        let mpi4py = Python.import_module("mpi4py")
        let MPI = mpi4py.MPI
        
        self.comm = MPI.COMM_WORLD
        self.rank = int(self.comm.Get_rank())
        self.size = int(self.comm.Get_size())
    
    fn barrier(self):
        """MPI barrier synchronization"""
        self.comm.Barrier()
    
    fn gather(self, data: PythonObject, root: Int = 0) -> PythonObject:
        """Gather data to root rank"""
        return self.comm.gather(data, root=root)
    
    fn broadcast(self, data: PythonObject, root: Int = 0) -> PythonObject:
        """Broadcast data from root rank"""
        return self.comm.bcast(data, root=root)
    
    fn allreduce(self, data: Float64) -> Float64:
        """All-reduce operation"""
        let result = self.comm.allreduce(data)
        return Float64(result)
    
    fn finalize(self):
        """Finalize MPI"""
        # MPI finalize happens automatically in mpi4py

# ============================================================================
# Configuration
# ============================================================================

struct Config:
    """CLI configuration parameters"""
    var testcases: DynamicVector[String]
    var suites: DynamicVector[String]
    var buffer_size: Int
    var iterations: Int
    var repeat: Int
    var duration: Int  # seconds
    var allocator_strategy: String
    var rich_output: Bool
    var samples_per_rack: Int
    var output_format: String
    
    fn __init__(inout self):
        self.testcases = DynamicVector[String]()
        self.suites = DynamicVector[String]()
        self.buffer_size = 512 * 1024 * 1024  # 512 MiB
        self.iterations = 16
        self.repeat = 1
        self.duration = 0
        self.allocator_strategy = "reuse"
        self.rich_output = False
        self.samples_per_rack = 5
        self.output_format = "text"
    
    fn parse_args(inout self, args: DynamicVector[String]):
        """Parse command line arguments"""
        var i = 1  # Skip program name
        while i < len(args):
            let arg = args[i]
            
            if arg == "-t" or arg == "--testcase":
                i += 1
                while i < len(args) and not args[i].startswith("-"):
                    self.testcases.append(args[i])
                    i += 1
                i -= 1
            
            elif arg == "-s" or arg == "--suite":
                i += 1
                while i < len(args) and not args[i].startswith("-"):
                    self.suites.append(args[i])
                    i += 1
                i -= 1
            
            elif arg == "-b" or arg == "--bufferSize":
                i += 1
                if i < len(args):
                    self.buffer_size = Self._parse_size(args[i])
            
            elif arg == "-i" or arg == "--iterations":
                i += 1
                if i < len(args):
                    self.iterations = int(args[i])
            
            elif arg == "-c" or arg == "--repeat":
                i += 1
                if i < len(args):
                    self.repeat = int(args[i])
            
            elif arg == "-d" or arg == "--duration":
                i += 1
                if i < len(args):
                    self.duration = int(args[i])
            
            elif arg == "-a" or arg == "--allocatorStrategy":
                i += 1
                if i < len(args):
                    self.allocator_strategy = args[i]
            
            elif arg == "-r" or arg == "--richOutput":
                self.rich_output = True
            
            elif arg == "--samplesPerRack":
                i += 1
                if i < len(args):
                    self.samples_per_rack = int(args[i])
            
            elif arg == "-l" or arg == "--listTestcases":
                self.list_testcases()
                sys.exit(0)
            
            elif arg == "-h" or arg == "--help":
                self.print_help()
                sys.exit(0)
            
            i += 1
    
    @staticmethod
    fn _parse_size(size_str: String) -> Int:
        """Parse size string (e.g., '512M', '1G')"""
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
    
    fn list_testcases(self):
        """List available testcases"""
        print("Available Testcases:")
        print("  Pairwise Tests:")
        print("    - pairwise_device_to_device_write_sm")
        print("    - pairwise_device_to_device_read_sm")
        print("    - pairwise_device_to_device_write_ce")
        print("    - pairwise_device_to_device_read_ce")
        print("    - pairwise_device_to_device_bidir_sm")
        print("  Bisect Tests:")
        print("    - bisect_device_to_device_write_sm")
        print("    - bisect_device_to_device_read_sm")
        print("    - bisect_device_to_device_bidir_sm")
        print("  GPU-to-Rack Tests:")
        print("    - gpu_to_rack_write_sm")
        print("    - gpu_to_rack_read_sm")
        print("  Fabric Stress Tests:")
        print("    - fabric_stress_write_sm")
        print("    - fabric_stress_read_sm")
        print("  Multicast Tests:")
        print("    - multicast_one_to_all")
        print("    - multicast_all_to_all")
        print("    - multicast_reduce")
        print("  EGM Tests (if available):")
        print("    - egm_pairwise_write")
        print("    - egm_pairwise_read")
    
    fn print_help(self):
        """Print help information"""
        print("NVloom CLI - MNNVL Fabric Testing Tool (Mojo Port)")
        print()
        print("Usage: nvloom_cli [OPTIONS]")
        print()
        print("Options:")
        print("  -t, --testcase NAME       Run specific testcase(s)")
        print("  -s, --suite NAME          Run test suite(s)")
        print("  -b, --bufferSize SIZE     Buffer size (default: 512M)")
        print("  -i, --iterations N        Number of iterations (default: 16)")
        print("  -c, --repeat N            Repeat each test N times")
        print("  -d, --duration SECONDS    Run tests for specified duration")
        print("  -a, --allocatorStrategy   Allocation strategy: unique/reuse/cudapool")
        print("  -r, --richOutput          Show detailed per-measurement data")
        print("  --samplesPerRack N        Samples per rack for gpu-to-rack (default: 5)")
        print("  -l, --listTestcases       List all available testcases")
        print("  -h, --help                Show this help message")
        print()
        print("Available Suites:")
        print("  - pairwise: O(n^2) pairwise testing")
        print("  - fabric-stress: All GPUs transferring simultaneously")
        print("  - gpu-to-rack: Linear-time rack bandwidth testing")
        print("  - rack-to-rack: Rack-level bisection bandwidth")
        print("  - all-to-one: Multiple GPUs to single GPU")
        print("  - multicast: Multicast operations")

# ============================================================================
# Test Execution
# ============================================================================

struct TestExecutor:
    """Executes tests based on configuration"""
    var config: Config
    var mpi: MPIWrapper
    var nvloom: NVloom
    
    fn __init__(inout self, config: Config, mpi: MPIWrapper):
        self.config = config
        self.mpi = mpi
        # Initialize NVloom with one GPU per MPI rank
        self.nvloom = NVloom(mpi.size)
        self.nvloom.buffer_size = config.buffer_size
        self.nvloom.iterations = config.iterations
    
    fn run(inout self):
        """Run configured tests"""
        var all_results = DynamicVector[TestResult]()
        
        # Run testcases
        for i in range(len(self.config.testcases)):
            let testcase = self.config.testcases[i]
            let result = self.run_testcase(testcase)
            all_results.append(result)
        
        # Run suites
        for i in range(len(self.config.suites)):
            let suite = self.config.suites[i]
            let suite_results = self.nvloom.run_suite(suite)
            for j in range(len(suite_results)):
                all_results.append(suite_results[j])
        
        # If no specific tests requested, run default
        if len(all_results) == 0:
            all_results = self.run_default_tests()
        
        # Gather results to root rank
        if self.mpi.rank == 0:
            self.print_results(all_results)
    
    fn run_testcase(inout self, name: String) -> TestResult:
        """Run a specific testcase by name"""
        # Parse testcase name to extract pattern, copy type, and engine
        var pattern = TestPattern.PAIRWISE
        var copy_type = CopyType.WRITE
        var engine = CopyEngine.SM
        
        if "pairwise" in name:
            pattern = TestPattern.PAIRWISE
        elif "bisect" in name:
            pattern = TestPattern.BISECT
        elif "gpu_to_rack" in name:
            pattern = TestPattern.GPU_TO_RACK
        elif "fabric_stress" in name:
            pattern = TestPattern.BISECT  # Use bisect for fabric stress
        
        if "read" in name:
            copy_type = CopyType.READ
        elif "bidir" in name:
            copy_type = CopyType.BIDIRECTIONAL
        
        if "ce" in name:
            engine = CopyEngine.CE
        
        # Synchronize before test
        self.mpi.barrier()
        
        # Run test with repeats or duration
        if self.config.duration > 0:
            return self.run_with_duration(pattern, copy_type, engine)
        else:
            return self.run_with_repeats(pattern, copy_type, engine)
    
    fn run_with_repeats(inout self, pattern: Int, copy_type: Int, 
                        engine: Int) -> TestResult:
        """Run test with specified number of repeats"""
        var best_result: TestResult = TestResult("", pattern)
        var best_bandwidth = 0.0
        
        for repeat in range(self.config.repeat):
            let result = self.nvloom.run_testcase(pattern, copy_type, engine)
            
            if result.summary_stats.mean_bandwidth > best_bandwidth:
                best_bandwidth = result.summary_stats.mean_bandwidth
                best_result = result
        
        return best_result
    
    fn run_with_duration(inout self, pattern: Int, copy_type: Int,
                         engine: Int) -> TestResult:
        """Run test for specified duration"""
        let start_time = now()
        let duration_ns = self.config.duration * 1_000_000_000
        
        var combined_result = TestResult("duration_test", pattern)
        var run_count = 0
        
        while (now() - start_time) < duration_ns:
            let result = self.nvloom.run_testcase(pattern, copy_type, engine)
            
            # Combine measurements
            for i in range(len(result.measurements)):
                combined_result.add_measurement(result.measurements[i])
            
            run_count += 1
        
        print("Completed", run_count, "runs in", self.config.duration, "seconds")
        return combined_result
    
    fn run_default_tests(inout self) -> DynamicVector[TestResult]:
        """Run default test suite"""
        print("Running default tests...")
        var results = DynamicVector[TestResult]()
        
        # Quick bisect test
        results.append(self.nvloom.run_testcase(
            TestPattern.BISECT, CopyType.WRITE, CopyEngine.SM
        ))
        
        # GPU-to-rack test if enough GPUs
        if self.mpi.size >= 16:
            results.append(self.nvloom.run_testcase(
                TestPattern.GPU_TO_RACK, CopyType.WRITE, CopyEngine.SM
            ))
        
        return results
    
    fn print_results(self, results: DynamicVector[TestResult]):
        """Print test results"""
        print("\n" + "="*60)
        print("NVloom Test Results")
        print("="*60)
        print("Configuration:")
        print("  GPUs:", self.mpi.size)
        print("  Buffer Size:", self.config.buffer_size // (1024*1024), "MB")
        print("  Iterations:", self.config.iterations)
        print()
        
        if self.config.output_format == "json":
            self.print_json_results(results)
        else:
            self.print_text_results(results)
    
    fn print_text_results(self, results: DynamicVector[TestResult]):
        """Print results in text format"""
        for i in range(len(results)):
            let result = results[i]
            print("-" * 40)
            print("Test:", result.test_name)
            print("Pattern:", Self._pattern_name(result.pattern))
            print("Measurements:", len(result.measurements))
            
            if self.config.rich_output:
                # Show individual measurements
                for j in range(min(10, len(result.measurements))):
                    let m = result.measurements[j]
                    print(f"  GPU {m.source_gpu} -> GPU {m.dest_gpu}: "
                          f"{m.bandwidth_gbps:.2f} GB/s")
                if len(result.measurements) > 10:
                    print("  ... and", len(result.measurements) - 10, "more")
            
            print("Summary Statistics:")
            print(f"  Mean:   {result.summary_stats.mean_bandwidth:.2f} GB/s")
            print(f"  Min:    {result.summary_stats.min_bandwidth:.2f} GB/s")
            print(f"  Max:    {result.summary_stats.max_bandwidth:.2f} GB/s")
            print(f"  Median: {result.summary_stats.median_bandwidth:.2f} GB/s")
    
    fn print_json_results(self, results: DynamicVector[TestResult]):
        """Print results in JSON format"""
        print("{")
        print('  "nvloom_results": {')
        print(f'    "num_gpus": {self.mpi.size},')
        print(f'    "buffer_size": {self.config.buffer_size},')
        print(f'    "iterations": {self.config.iterations},')
        print('    "tests": [')
        
        for i in range(len(results)):
            let result = results[i]
            print('      {')
            print(f'        "name": "{result.test_name}",')
            print(f'        "pattern": "{Self._pattern_name(result.pattern)}",')
            print(f'        "measurements": {len(result.measurements)},')
            print('        "summary": {')
            print(f'          "mean": {result.summary_stats.mean_bandwidth},')
            print(f'          "min": {result.summary_stats.min_bandwidth},')
            print(f'          "max": {result.summary_stats.max_bandwidth},')
            print(f'          "median": {result.summary_stats.median_bandwidth}')
            print('        }')
            print('      }' + (',' if i < len(results) - 1 else ''))
        
        print('    ]')
        print('  }')
        print('}')
    
    @staticmethod
    fn _pattern_name(pattern: Int) -> String:
        """Get pattern name from enum value"""
        if pattern == TestPattern.PAIRWISE:
            return "pairwise"
        elif pattern == TestPattern.BISECT:
            return "bisect"
        elif pattern == TestPattern.GPU_TO_RACK:
            return "gpu-to-rack"
        elif pattern == TestPattern.RACK_TO_RACK:
            return "rack-to-rack"
        elif pattern == TestPattern.FABRIC_STRESS:
            return "fabric-stress"
        else:
            return "unknown"

# ============================================================================
# Main Entry Point
# ============================================================================

fn main() raises:
    """Main entry point for NVloom CLI"""
    # Initialize MPI
    var mpi = MPIWrapper()
    
    if mpi.rank == 0:
        print("NVloom Mojo - MNNVL Fabric Testing Tool")
        print(f"Running with {mpi.size} MPI ranks (GPUs)")
        print()
    
    # Parse command line arguments
    var args = DynamicVector[String]()
    for i in range(len(sys.argv)):
        args.append(sys.argv[i])
    
    var config = Config()
    config.parse_args(args)
    
    # Execute tests
    var executor = TestExecutor(config, mpi)
    executor.run()
    
    # Cleanup
    executor.nvloom.cleanup()
    mpi.finalize()
    
    if mpi.rank == 0:
        print("\nTests complete!")
