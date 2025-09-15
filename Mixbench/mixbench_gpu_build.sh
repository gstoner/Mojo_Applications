#!/bin/bash

# Build script for Mixbench-Mojo GPU version
# Builds both CUDA kernels and Mojo host code for NVIDIA B100

set -e

echo "=========================================="
echo "Building Mixbench-Mojo GPU (B100 Enabled)"
echo "=========================================="

# Check for required tools
echo "Checking prerequisites..."

# Check Mojo
if ! command -v mojo &> /dev/null; then
    echo "❌ Error: Mojo compiler not found!"
    echo "Please install the Mojo SDK from: https://www.modular.com/mojo"
    exit 1
fi

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: NVIDIA CUDA compiler (nvcc) not found!"
    echo "Please install CUDA Toolkit 12.0 or later for B100 support"
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check for nvidia-smi (driver)
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU detection may not work."
else
    echo "✅ NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | head -1
fi

echo ""
echo "Tool versions:"
mojo --version
nvcc --version | grep "release"

# Set build configuration
BUILD_DIR="build_gpu"
CUDA_ARCH="90"  # B100 Blackwell architecture
CUDA_FLAGS="-O3 -use_fast_math -Xptxas -O3 -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
MOJO_FLAGS="-O3"

# Detect system and set paths
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Building for Linux..."
    CUDA_LIB_PATH="/usr/local/cuda/lib64"
    CUDA_INC_PATH="/usr/local/cuda/include"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Building for macOS..."
    echo "⚠️  Warning: CUDA on macOS is not officially supported for recent versions"
    CUDA_LIB_PATH="/usr/local/cuda/lib"
    CUDA_INC_PATH="/usr/local/cuda/include"
else
    echo "Building for Windows/Other..."
    # Adjust paths as needed for Windows
    CUDA_LIB_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64"
    CUDA_INC_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "=========================================="
echo "Building CUDA Kernels"
echo "=========================================="

# Compile CUDA kernels to object file
echo "Compiling CUDA kernels for B100..."
echo "CUDA flags: $CUDA_FLAGS"

nvcc $CUDA_FLAGS -I"$CUDA_INC_PATH" -c ../mixbench_kernels.cu -o mixbench_kernels.o

if [ $? -eq 0 ]; then
    echo "✅ CUDA kernels compiled successfully"
else
    echo "❌ CUDA kernel compilation failed"
    exit 1
fi

# Create shared library from CUDA kernels
echo "Creating CUDA kernel shared library..."
nvcc $CUDA_FLAGS -shared mixbench_kernels.o -o libmixbench_cuda.so -L"$CUDA_LIB_PATH" -lcudart

if [ $? -eq 0 ]; then
    echo "✅ CUDA shared library created: libmixbench_cuda.so"
else
    echo "❌ CUDA shared library creation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Building Mojo Host Code"
echo "=========================================="

# Compile Mojo GPU code
echo "Compiling Mojo GPU host code..."
echo "Mojo flags: $MOJO_FLAGS"

# Copy source to build directory for proper linking
cp ../mixbench_gpu.mojo .

# Build with library linking
mojo build $MOJO_FLAGS mixbench_gpu.mojo -o mixbench-mojo-gpu \
    -I "$CUDA_INC_PATH" \
    -L . -L "$CUDA_LIB_PATH" \
    -lmixbench_cuda -lcudart

if [ $? -eq 0 ]; then
    echo "✅ Mojo GPU code compiled successfully"
else
    echo "❌ Mojo GPU compilation failed"
    echo "Make sure CUDA libraries are properly installed and accessible"
    exit 1
fi

# Make executable
chmod +x mixbench-mojo-gpu

cd ..

echo ""
echo "=========================================="
echo "Build Summary"
echo "=========================================="
echo "✅ Build completed successfully!"
echo ""
echo "Generated files:"
echo "  📁 $BUILD_DIR/"
echo "    📄 mixbench_kernels.o     - CUDA kernel object file"
echo "    📚 libmixbench_cuda.so    - CUDA kernel library"
echo "    🚀 mixbench-mojo-gpu      - Main executable"
echo ""
echo "To run the GPU benchmark:"
echo "  cd $BUILD_DIR"
echo "  export LD_LIBRARY_PATH=.:$CUDA_LIB_PATH:\$LD_LIBRARY_PATH"
echo "  ./mixbench-mojo-gpu"
echo ""

# GPU detection and recommendations
echo "=========================================="
echo "GPU Detection & Recommendations"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo "Available NVIDIA GPUs:"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free --format=csv,noheader
    
    # Check for B100 specifically
    B100_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -c "B100" || echo "0")
    H100_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -c "H100" || echo "0")
    
    if [ "$B100_COUNT" -gt 0 ]; then
        echo ""
        echo "🎉 NVIDIA B100 GPU detected! Optimal configuration:"
        echo "  - FP8 precision benchmarks: ✅ Supported"
        echo "  - 4th-gen Tensor Cores: ✅ Supported" 
        echo "  - HBM3e memory: ✅ Supported"
        echo "  - Recommended buffer size: 4-8GB"
    elif [ "$H100_COUNT" -gt 0 ]; then
        echo ""
        echo "🚀 NVIDIA H100 GPU detected! Good compatibility:"
        echo "  - FP8 precision benchmarks: ✅ Supported"
        echo "  - 4th-gen Tensor Cores: ✅ Supported"
        echo "  - Recommended buffer size: 2-4GB"
    else
        echo ""
        echo "ℹ️  Other NVIDIA GPU detected:"
        echo "  - FP8 precision benchmarks: ❓ May not be supported"
        echo "  - Tensor Cores: ❓ Depends on architecture"
        echo "  - Will fall back to FP32/FP64 benchmarks"
        echo "  - Recommended buffer size: 512MB-2GB"
    fi
else
    echo "⚠️  Cannot detect GPU. Please ensure:"
    echo "  1. NVIDIA drivers are installed"
    echo "  2. GPU is properly connected"
    echo "  3. CUDA runtime is working"
fi

echo ""
echo "Performance tuning tips:"
echo "  🔧 Set GPU to performance mode:"
echo "     sudo nvidia-smi -pm 1"
echo "     sudo nvidia-smi -ac <mem_clock>,<gpu_clock>"
echo ""
echo "  🔧 For best results, run with exclusive compute mode:"
echo "     sudo nvidia-smi -c 3"
echo ""
echo "  🔧 Monitor GPU during benchmark:"
echo "     nvidia-smi dmon -s pucvmet -d 1"
echo ""
echo "Ready to benchmark! 🚀"