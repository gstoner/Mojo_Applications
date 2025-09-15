# Mixbench-Mojo

A Mojo language port of the [Mixbench GPU benchmark tool](https://github.com/ekondis/mixbench) by Elias Konstantinidis.

## Overview

Mixbench-Mojo evaluates the performance bounds of CPUs (and potentially GPUs in the future) on mixed operational intensity kernels. The tool varies the ratio of compute operations to memory accesses to identify the optimal balance point for different hardware configurations.

This Mojo port brings the powerful benchmarking capabilities of the original Mixbench to the Mojo ecosystem, leveraging Mojo's high-performance parallel computing features and zero-cost abstractions.

## Key Features

- **Multi-precision benchmarking**: Tests single precision (F32), double precision (F64), and integer (I32) operations
- **Operational intensity scaling**: Varies compute-to-memory ratios from memory-bound to compute-bound workloads  
- **Parallel execution**: Uses Mojo's `parallelize` for optimal CPU utilization
- **Performance metrics**: Reports FLOPS/IOPS, bandwidth utilization, and operational intensity
- **CSV output**: Compatible data format with original Mixbench for analysis and plotting

## Building and Running

### Prerequisites

- Mojo SDK (latest version recommended)
- System with multiple CPU cores for parallel benchmarking

### Building

```bash
# Compile the Mojo benchmark
mojo build mixbench.mojo -o mixbench-mojo

# Or run directly
mojo run mixbench.mojo
```

### Running

```bash
# Run with default settings (256MB buffer)
./mixbench-mojo

# The tool currently uses hardcoded buffer size, but can be easily modified
# to accept command line arguments similar to the original
```

## Sample Output

```
mixbench-mojo (v1.0-mojo-port)
A Mojo port of the mixbench GPU/CPU benchmark tool
Original mixbench by Elias Konstantinidis
Mojo port by Claude (Anthropic)

------------------------ Device specifications ------------------------
Device: Mojo CPU
Total memory: 1024 MB
Compute units: 16
SIMD width: 8
-----------------------------------------------------------------------
Buffer size: 256 MB
Trade-off type: compute with memory (Mojo parallel)
Elements per thread: 8
Fusion degree: 4

----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------
Experiment ID, Single Precision ops,,,, Double precision ops,,,, Integer operations,,,
Compute iters, Flops/byte, ex.time, GFLOPS, GB/sec, Flops/byte, ex.time, GFLOPS, GB/sec, Iops/byte, ex.time, GIOPS, GB/sec
0, 0.250, 12.34, 85.42, 341.68, 0.125, 15.63, 42.04, 336.35, 0.250, 11.32, 88.58, 354.30
1, 0.750, 13.45, 245.34, 327.12, 0.375, 18.23, 110.69, 295.18, 0.750, 12.87, 248.30, 331.07
...
```

## Performance Characteristics

The Mojo port demonstrates several performance advantages:

1. **Zero-cost abstractions**: Mojo's compile-time optimizations eliminate runtime overhead
2. **Parallel execution**: Automatic parallelization across CPU cores
3. **SIMD optimization**: Vectorized operations where beneficial
4. **Memory efficiency**: Optimized tensor operations with minimal allocation overhead

## Architectural Differences from Original

While maintaining compatibility with the original Mixbench methodology, the Mojo port includes several adaptations:

### CPU-Focused Design
- Optimized for multi-core CPU execution rather than GPU kernels
- Uses Mojo's `parallelize` instead of CUDA thread blocks
- Adapts memory access patterns for CPU cache hierarchies

### Modern Language Features
- Compile-time parameters for kernel specialization
- Strong type system with zero-cost abstractions  
- Memory-safe tensor operations
- Generic programming with specialization

### Extensibility
- Modular design for adding new data types
- Easy integration with Mojo's ML/AI ecosystem
- Potential for future GPU backend integration

## Technical Implementation

### Kernel Design

The benchmark kernels follow this pattern:

```mojo
@parameter
fn benchmark_kernel_float32[compute_iters: Int](
    inout data: Tensor[DType.float32], 
    scalar_val: Float32,
    size: Int
) -> Float64:
    # Parallel execution across data
    @parameter
    fn compute_intensive_work(i: Int):
        var value = data[i % size]
        
        # Parameterized compute intensity
        @parameter
        for _ in range(compute_iters):
            value = value * scalar_val + scalar_val
            
        data[i % size] = value
    
    parallelize[compute_intensive_work](size)
```

### Key Optimizations

1. **Compile-time specialization**: Each compute intensity level is specialized at compile time
2. **Memory layout optimization**: Tensor operations optimized for CPU cache lines  
3. **Parallel decomposition**: Work distributed across available CPU cores
4. **Measurement accuracy**: Multiple runs with warmup for stable timing

## Extending the Benchmark

### Adding New Data Types

```mojo
# Add new tensor type (example: F16)
fn run_half_precision_benchmark(compute_iters: Int, buffer_size: Int) -> BenchmarkResult:
    let data_size = buffer_size // sizeof[DType.float16]()
    var data = Tensor[DType.float16](TensorShape(data_size))
    # ... implementation
```

### Custom Compute Patterns

```mojo
# Example: Add transcendental operations
@parameter
fn benchmark_kernel_transcendental[compute_iters: Int](
    inout data: Tensor[DType.float32]
) -> Float64:
    @parameter
    fn work(i: Int):
        var value = data[i]
        @parameter
        for _ in range(compute_iters):
            value = sqrt(value * value + 1.0)
        data[i] = value
    # ...
```

## Performance Analysis

The results can be analyzed to understand:

1. **Memory vs Compute Bound Regions**: Identify where performance plateaus
2. **Optimal Operational Intensity**: Find the sweet spot for your hardware
3. **Scaling Characteristics**: How performance scales with problem size
4. **Architecture Comparison**: Compare different CPU architectures

## Future Enhancements

- [ ] GPU backend support when Mojo GPU capabilities mature
- [ ] Command-line argument parsing
- [ ] Additional precision types (BF16, etc.)
- [ ] Custom memory access patterns
- [ ] Integration with Mojo ML/AI frameworks
- [ ] Performance visualization tools
- [ ] Multi-device benchmarking

## Citation

If you use this benchmark tool for research, please cite both the original Mixbench papers and acknowledge this Mojo port:

**Original Mixbench:**
```
Elias Konstantinidis, Yiannis Cotronis,
"A quantitative roofline model for GPU kernel performance estimation using micro-benchmarks and hardware metric profiling",
Journal of Parallel and Distributed Computing, Volume 107, September 2017, Pages 37-56, ISSN 0743-7315,
https://doi.org/10.1016/j.jpdc.2017.04.002

Konstantinidis, E., Cotronis, Y.,
"A Practical Performance Model for Compute and Memory Bound GPU Kernels",
Parallel, Distributed and Network-Based Processing (PDP), 2015 23rd Euromicro International Conference on,
vol., no., pp.651-658, 4-6 March 2015, doi: 10.1109/PDP.2015.51
```

**This Mojo Port:**
```
Claude (Anthropic AI Assistant),
"Mixbench-Mojo: A Mojo Language Port of the Mixbench Benchmark Suite",
2025, Based on original work by Elias Konstantinidis
```

## Comparison with Original Mixbench

| Feature | Original CUDA/OpenCL | Mojo Port |
|---------|---------------------|-----------|
| **Target Platform** | NVIDIA/AMD GPUs | Multi-core CPUs |
| **Parallelism** | CUDA threads/OpenCL work-items | Mojo parallelize |
| **Memory Model** | GPU global memory | System RAM + CPU caches |
| **Precision Support** | FP32, FP64, FP16, INT32 | FP32, FP64, INT32 |
| **Compilation** | nvcc/clang | Mojo compiler |
| **Performance Focus** | GPU throughput | CPU parallel efficiency |

## Benchmark Interpretation Guide

### Understanding the Output

The CSV output provides four key metrics for each operational intensity level:

1. **Flops/byte (or Iops/byte)**: Operational intensity - higher values indicate more compute per memory access
2. **ex.time**: Execution time in milliseconds
3. **GFLOPS/GIOPS**: Computational throughput in billions of operations per second
4. **GB/sec**: Memory bandwidth utilization

### Performance Regions

**Memory-Bound Region (Low Operational Intensity):**
- Performance limited by memory bandwidth
- FLOPS increase linearly with operational intensity
- Memory bandwidth utilization is high

**Compute-Bound Region (High Operational Intensity):**
- Performance limited by computational throughput
- FLOPS plateau at peak compute performance
- Memory bandwidth utilization decreases

**Transition Point:**
- The "knee" in the performance curve
- Optimal balance between compute and memory operations
- Architecture-dependent sweet spot

### Analysis Tips

1. **Plot the results**: Create FLOPS vs Operational Intensity graphs
2. **Identify bottlenecks**: Find where performance saturates
3. **Compare architectures**: Run on different CPU types
4. **Optimize applications**: Target the identified optimal operational intensity

## Advanced Usage

### Custom Buffer Sizes

Modify the `DEFAULT_BUFFER_SIZE_MB` constant or add command-line argument parsing:

```mojo
# In main function
let buffer_size_mb = 64  # Use smaller buffer for L3 cache fitting
run_mixed_benchmark_suite(buffer_size_mb)
```

### Varying Thread Count

Control parallelism by modifying the parallelize call:

```mojo
# Limit to specific number of threads
parallelize[compute_intensive_work](size, workers=8)
```

### Custom Compute Patterns

Add domain-specific operations:

```mojo
@parameter
fn benchmark_kernel_ml_ops[compute_iters: Int](
    inout data: Tensor[DType.float32],
    size: Int
) -> Float64:
    # Example: ML-specific operations like ReLU, sigmoid, etc.
    @parameter
    fn ml_work(i: Int):
        var value = data[i % size]
        @parameter
        for _ in range(compute_iters):
            # ReLU activation
            value = max(0.0, value * 1.01 + 0.1)
        data[i % size] = value
    
    parallelize[ml_work](size)
    # ... timing logic
```

## Troubleshooting

### Common Issues

**Compilation Errors:**
- Ensure you have the latest Mojo SDK
- Check that all imports are available in your Mojo version

**Performance Anomalies:**
- System thermal throttling can affect results
- Background processes may interfere with measurements
- Run multiple times for statistical significance

**Memory Issues:**
- Reduce buffer size for systems with limited RAM
- Monitor system memory usage during benchmarks

### Platform-Specific Notes

**Linux:**
- May achieve better performance due to process scheduling
- Consider setting CPU affinity for consistent results

**macOS:**
- Performance may vary with thermal management
- Monitor Activity Monitor during execution

**Windows:**
- Antivirus software may interfere with timing
- Consider running in high-performance power mode

## Contributing

This Mojo port is designed to be extensible and welcomes contributions:

1. **Additional data types**: BF16, INT64, custom numeric types
2. **GPU backend**: When Mojo GPU support becomes available
3. **Optimization improvements**: Better SIMD utilization, cache optimization
4. **Analysis tools**: Plotting utilities, performance modeling
5. **Platform support**: ARM, RISC-V, specialized accelerators

## License

This Mojo port maintains compatibility with the original Mixbench licensing terms. Please refer to the original repository for detailed license information.

## Acknowledgments

- **Elias Konstantinidis**: Original Mixbench creator and methodology
- **Mojo Team**: For creating an excellent high-performance language
- **Research Community**: For the foundational work in performance modeling and roofline analysis

## Links

- [Original Mixbench Repository](https://github.com/ekondis/mixbench)
- [Mojo Programming Language](https://www.modular.com/mojo)
- [Roofline Performance Model](https://en.wikipedia.org/wiki/Roofline_model)
- [Performance Analysis Papers](https://scholar.google.com/scholar?q=roofline+model+performance+analysis)

---

*This Mojo port demonstrates the power of modern systems programming languages in creating high-performance benchmarking tools while maintaining scientific rigor and reproducibility.*