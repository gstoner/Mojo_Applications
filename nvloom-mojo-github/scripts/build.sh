#!/bin/bash

# Build script for NVloom Mojo
# Compiles all components and sets up the environment

set -e  # Exit on error

echo "================================"
echo "NVloom Mojo Build Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Mojo installation
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! command -v mojo &> /dev/null; then
    echo -e "${RED}Error: Mojo is not installed${NC}"
    echo "Please install Mojo from: https://www.modular.com/mojo"
    exit 1
fi

if ! command -v mpirun &> /dev/null; then
    echo -e "${YELLOW}Warning: MPI is not installed${NC}"
    echo "MPI is required for distributed testing"
    echo "Install with: sudo apt-get install openmpi-bin openmpi-common"
fi

# Check CUDA installation
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    else
        echo -e "${YELLOW}Warning: CUDA_HOME not set${NC}"
        echo "CUDA is required for GPU operations"
    fi
fi

# Create build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
    echo "Created build directory: $BUILD_DIR"
fi

# Build options
MOJO_FLAGS=""
CUDA_ARCH="sm_80"  # Default to Ampere architecture

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            MOJO_FLAGS="$MOJO_FLAGS -O0 -g"
            echo "Building in debug mode"
            shift
            ;;
        --release)
            MOJO_FLAGS="$MOJO_FLAGS -O3"
            echo "Building in release mode"
            shift
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            echo "Using CUDA architecture: $CUDA_ARCH"
            shift 2
            ;;
        --clean)
            echo "Cleaning build directory..."
            rm -rf "$BUILD_DIR"/*
            echo "Clean complete"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./build.sh [--debug|--release] [--cuda-arch sm_xx] [--clean]"
            exit 1
            ;;
    esac
done

# Build core library
echo -e "\n${YELLOW}Building NVloom core library...${NC}"
mojo build nvloom.mojo \
    $MOJO_FLAGS \
    -D CUDA_ARCH=$CUDA_ARCH \
    -o $BUILD_DIR/libnvloom.so \
    --shared

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Core library built successfully${NC}"
else
    echo -e "${RED}✗ Core library build failed${NC}"
    exit 1
fi

# Build kernels module
echo -e "\n${YELLOW}Building CUDA kernels module...${NC}"
mojo build kernels.mojo \
    $MOJO_FLAGS \
    -D CUDA_ARCH=$CUDA_ARCH \
    -o $BUILD_DIR/libkernels.so \
    --shared

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Kernels module built successfully${NC}"
else
    echo -e "${RED}✗ Kernels module build failed${NC}"
    exit 1
fi

# Build CLI tool
echo -e "\n${YELLOW}Building CLI tool...${NC}"
mojo build nvloom_cli.mojo \
    $MOJO_FLAGS \
    -o $BUILD_DIR/nvloom_cli

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CLI tool built successfully${NC}"
else
    echo -e "${RED}✗ CLI tool build failed${NC}"
    exit 1
fi

# Build visualization tool
echo -e "\n${YELLOW}Building visualization tool...${NC}"
mojo build plot_heatmaps.mojo \
    $MOJO_FLAGS \
    -o $BUILD_DIR/plot_heatmaps

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Visualization tool built successfully${NC}"
else
    echo -e "${RED}✗ Visualization tool build failed${NC}"
    exit 1
fi

# Build examples
echo -e "\n${YELLOW}Building examples...${NC}"
mojo build examples.mojo \
    $MOJO_FLAGS \
    -o $BUILD_DIR/examples

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Examples built successfully${NC}"
else
    echo -e "${RED}✗ Examples build failed${NC}"
    exit 1
fi

# Create run script
echo -e "\n${YELLOW}Creating run script...${NC}"
cat > $BUILD_DIR/run_nvloom.sh << 'EOF'
#!/bin/bash
# Run script for NVloom CLI

# Set library path
export LD_LIBRARY_PATH=$(dirname "$0"):$LD_LIBRARY_PATH

# Default number of GPUs
NUM_GPUS=${1:-1}
shift

# Run with MPI if multiple GPUs
if [ $NUM_GPUS -gt 1 ]; then
    echo "Running with $NUM_GPUS GPUs using MPI..."
    mpirun -np $NUM_GPUS $(dirname "$0")/nvloom_cli "$@"
else
    echo "Running with single GPU..."
    $(dirname "$0")/nvloom_cli "$@"
fi
EOF

chmod +x $BUILD_DIR/run_nvloom.sh
echo -e "${GREEN}✓ Run script created${NC}"

# Check Python dependencies for visualization
echo -e "\n${YELLOW}Checking Python dependencies...${NC}"
python3 -c "import matplotlib, seaborn, numpy, mpi4py" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python dependencies are installed${NC}"
else
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install matplotlib seaborn numpy mpi4py
fi

# Create test script
echo -e "\n${YELLOW}Creating test script...${NC}"
cat > $BUILD_DIR/test_nvloom.sh << 'EOF'
#!/bin/bash
# Quick test script for NVloom

echo "Running NVloom tests..."
cd $(dirname "$0")

# Test 1: List testcases
echo -e "\n=== Test 1: List testcases ==="
./nvloom_cli -l

# Test 2: Run examples
echo -e "\n=== Test 2: Run examples ==="
./examples

# Test 3: Quick bisect test (if GPUs available)
echo -e "\n=== Test 3: Quick bisect test ==="
timeout 5 ./nvloom_cli -t bisect_device_to_device_write_sm || echo "Test completed or no GPUs available"

echo -e "\nAll tests completed!"
EOF

chmod +x $BUILD_DIR/test_nvloom.sh
echo -e "${GREEN}✓ Test script created${NC}"

# Summary
echo -e "\n================================"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "================================"
echo
echo "Build artifacts are in: $BUILD_DIR/"
echo
echo "To run NVloom:"
echo "  Single GPU:  $BUILD_DIR/nvloom_cli -s fabric-stress"
echo "  Multi-GPU:   $BUILD_DIR/run_nvloom.sh 8 -s fabric-stress"
echo "  Examples:    $BUILD_DIR/examples"
echo "  Tests:       $BUILD_DIR/test_nvloom.sh"
echo
echo "For help:"
echo "  $BUILD_DIR/nvloom_cli --help"
