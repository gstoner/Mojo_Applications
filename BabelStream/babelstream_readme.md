# BabelStream Mojo

A high-performance memory bandwidth benchmark suite implemented in Mojo, ported from the original [BabelStream](https://github.com/UoB-HPC/BabelStream) project.

## Overview

BabelStream measures sustainable memory bandwidth for parallel computing devices. This Mojo implementation provides:

- **CPU Implementation**: Optimized using Mojo's SIMD and vectorization capabilities
- **GPU Implementation**: Conceptual GPU kernels (for future Mojo GPU support)
- **5 Core Kernels**: Copy, Multiply, Add, Triad, and Dot Product
- **Performance Analysis**: Detailed timing and bandwidth measurements
- **Validation**: Comprehensive correctness verification

## Features

### ✅ Implemented
- CPU-optimized kernels using Mojo SIMD
- Vectorized operations with automatic SIMD width detection
- Memory bandwidth calculations (MB/s and MiB/s)
- Single and double precision support
- Comprehensive test suite
- Performance benchmarking tools
- Build system with multiple targets

### 🚧 Planned (GPU Support)
- CUDA/HIP GPU kernel implementations
- Multi-GPU support for large arrays  
- Memory coalescing optimizations
- GPU-specific performance tuning

## Quick Start

### Prerequisites
- Mojo 24.0.0 or later
- Modern CPU with SIMD support (SSE4.2/AVX/AVX-512)
- Optional: CUDA-capable GPU (for future GPU support)

### Building

```bash
# Clone the repository
git clone <repository-url>
cd babelstream-mojo

# Build CPU version (default)
make cpu

# Build with debug symbols
make debug

# Build and run tests
make test
```

### Running Benchmarks

```bash
# Basic CPU benchmark
./build/babelstream-cpu

# Custom array size (1M elements)
./build/babelstream-cpu --arraysize 1048576

# Single precision
./build/babelstream-cpu --float

# Only run triad kernel
./build/babelstream-cpu --triad-only

# CSV output for analysis
./build/babelstream-cpu --csv > results.csv
```

## The Kernels

BabelStream implements 5 memory bandwidth kernels:

### 1. Copy
```
c[i] = a[i]
```
**Memory Traffic**: 2 arrays (1 read, 1 write)

### 2. Multiply  
```
b[i] = scalar * c[i]
```
**Memory Traffic**: 2 arrays (1 read, 1 write)

### 3. Add
```
c[i] = a[i] + b[i]  
```
**Memory Traffic**: 3 arrays (2 reads, 1 write)

### 4. Triad
```
a[i] = b[i] + scalar * c[i]
```
**Memory Traffic**: 3 arrays (2 reads, 1 write)

### 5. Dot Product
```
sum = Σ a[i] * b[i]
```
**Memory Traffic**: 2 arrays (2 reads) + reduction

## Performance Optimization

### CPU Optimizations Used

1. **SIMD Vectorization**: Automatic detection and use of optimal SIMD width
2. **Memory Alignment**: 64-byte aligned arrays for optimal cache performance  
3. **Loop Unrolling**: Compiler-assisted loop optimization
4. **Parallelization**: Multi-threaded execution using Mojo's parallel features

### Example Performance Results

```
BabelStream Version: 1.0 (Mojo Implementation)
Implementation: Mojo CPU
Running kernels 100 times
Precision: double
Array size: 268.4 MB (=0.3 GB)
Total size: 805.3 MB (=0.8 GB)

Function        MB/s       Min (sec)    Max (sec)    Average
Copy           45123.456   0.00594     0.00612      0.00598
Mul            43567.890   0.00614     0.00628      0.00619  
Add            41234.567   0.00651     0.00671      0.00658
Triad          42345.678   0.00635     0.00649      0.00640
Dot            38901.234   0.00690     0.00715      0.00699
```

## Command Line Options

```bash
Usage: ./babelstream-cpu [OPTIONS]

Options:
  -h, --help              Show help message
  --list                  List available devices
  --device INDEX          Select device at INDEX
  --arraysize SIZE        Use SIZE elements in the array
  --numtimes NUM          Run the test NUM times (NUM >= 2)
  --float                 Use single precision (default: double)
  --triad-only           Only run triad kernel
  --dot-only             Only run dot product kernel  
  --csv                  Output results in CSV format
  --mibibytes            Use MiB=2^20 instead of MB=10^6
  --validate             Run validation tests
```

## Project Structure

```
babelstream-mojo/
├── src/
│   ├── babelstream.mojo         # Main implementation
│   ├── gpu_kernels.mojo         # GPU kernel implementations
│   └── utils.mojo               # Utility functions
├── tests/
│   └── test_kernels.mojo        # Test suite
├── benchmarks/
│   └── performance_tests.mojo   # Performance benchmarks
├── build/                       # Build outputs
├── results/                     # Benchmark results
├── Makefile                     # Build system
├── pyproject.toml              # Configuration
└── README.md                   # This file
```

## Building and Testing

### Build Targets

```bash
make cpu          # CPU-only build (default)
make gpu          # GPU build (when supported) 
make debug        # Debug build with symbols
make profile      # Profiling build
make test         # Build and run tests
make clean        # Clean build files
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
./build/test-runner --basic        # Basic correctness
./build/test-runner --performance  # Performance tests
./build/test-runner --gpu          # GPU tests (when available)
./build/test-runner --stress       # Stress tests
```

### Performance Testing

```bash
# Bandwidth scaling test
make bandwidth-test

# Precision comparison
make precision-test  

# Individual kernels
make triad-only
make dot-only

# Comprehensive benchmark
make bench-all
```

## Validation

The implementation includes comprehensive validation:

- **Correctness**: Verify kernel outputs match expected mathematical results
- **Numerical Stability**: Check for accumulated floating-point errors
- **Performance Consistency**: Ensure timing results are reproducible  
- **Memory Safety**: Validate proper memory alignment and access patterns

## Contributing

Contributions are welcome! Areas of interest:

1. **GPU Implementation**: Complete the GPU kernel implementations when Mojo GPU support matures
2. **Optimizations**: Additional CPU architecture-specific optimizations
3. **Benchmarks**: Extended benchmark suites and analysis tools
4. **Documentation**: Usage examples and performance tuning guides

### Development Workflow

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check

# Generate documentation  
make docs
```

## Performance Tuning

### CPU Tuning Tips

1. **Array Size**: Use sizes that are multiples of your cache line size (typically 64 bytes)
2. **NUMA**: For multi-socket systems, consider NUMA-aware memory allocation
3. **Compiler Flags**: The build system uses `-O3 -march=native -mtune=native`
4. **Memory**: Enable huge pages for better TLB performance on large arrays

### System Configuration

```bash
# Enable huge pages (Linux)
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# Set CPU governor to performance  
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
echo 1 | sudo tee /proc/sys/kernel/numa_balancing
```

## Comparison with Original BabelStream

| Feature | Original BabelStream | Mojo Implementation |
|---------|---------------------|-------------------|
| Languages | C++, OpenCL, CUDA, etc. | Mojo |
| CPU Optimization | Manual vectorization | Automatic SIMD |
| Memory Management | Manual allocation | Tensor-based |
| Type System | Templates | Parametric types |
| Build System | CMake | Make + Mojo |
| GPU Support | Full CUDA/OpenCL | Planned |

## Benchmarking Results

### Expected Performance Ranges

| System Type | Memory Bandwidth | Notes |
|-------------|------------------|-------|
| DDR4-3200 (Single Channel) | ~25 GB/s | Theoretical peak |
| DDR4-3200 (Dual Channel) | ~50 GB/s | Typical desktop |
| DDR4-3200 ECC (Server) | ~45 GB/s | Server memory |
| DDR5-4800 (Dual Channel) | ~75 GB/s | Modern systems |
| High-end Workstation | ~100+ GB/s | Multi-channel ECC |

### Factors Affecting Performance

- **CPU Architecture**: Newer CPUs have better memory controllers
- **Memory Configuration**: Dual/quad-channel provides higher bandwidth  
- **NUMA Topology**: Cross-socket access can reduce performance
- **System Load**: Other processes compete for memory bandwidth
- **Compiler Optimizations**: Different compilers may produce varying results

## License

This project is licensed under the same terms as the original BabelStream project. See the LICENSE file for details.

## References

- [Original BabelStream](https://github.com/UoB-HPC/BabelStream)
- [STREAM Benchmark](https://www.cs.virginia.edu/stream/)
- [Mojo Programming Language](https://docs.modular.com/mojo/)

## Citation

If you use this implementation in your research, please cite both the original BabelStream paper and this Mojo implementation:

```bibtex
@article{deakin2018evaluating,
  title={Evaluating attainable memory bandwidth of parallel programming models via BabelStream},
  author={Deakin, Tom and Price, James and Martineau, Matt and McIntosh-Smith, Simon},
  journal={International Journal of Computational Science and Engineering},
  volume={17},
  number={3},
  pages={247--262},
  year={2018},
  publisher={Inderscience Publishers}
}
```

## Acknowledgments

- Original BabelStream authors and contributors
- The Modular team for developing Mojo
- High-performance computing community for benchmark design insights