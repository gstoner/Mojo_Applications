#!/bin/bash

# Setup script for NVloom-Mojo
# Installs dependencies and configures the environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}NVloom-Mojo Setup Script${NC}"
echo -e "${BLUE}================================${NC}"
echo

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo -e "${YELLOW}Detected OS: $OS${NC}"

# Check for Mojo
echo -e "\n${YELLOW}Checking for Mojo...${NC}"
if command_exists mojo; then
    echo -e "${GREEN}✓ Mojo is installed$(NC)"
    mojo --version
else
    echo -e "${RED}✗ Mojo not found${NC}"
    echo "Please install Mojo from: https://www.modular.com/mojo"
    echo ""
    echo "Quick install:"
    echo "  curl https://get.modular.com | sh"
    echo "  modular install mojo"
    exit 1
fi

# Check for CUDA
echo -e "\n${YELLOW}Checking for CUDA...${NC}"
if [ -n "$CUDA_HOME" ] || [ -d "/usr/local/cuda" ]; then
    echo -e "${GREEN}✓ CUDA found${NC}"
    if [ -z "$CUDA_HOME" ]; then
        export CUDA_HOME=/usr/local/cuda
    fi
    echo "  CUDA_HOME: $CUDA_HOME"
    
    # Check CUDA version
    if [ -f "$CUDA_HOME/version.txt" ]; then
        cat "$CUDA_HOME/version.txt"
    elif command_exists nvcc; then
        nvcc --version | grep "release"
    fi
else
    echo -e "${YELLOW}⚠ CUDA not found${NC}"
    echo "CUDA is required for GPU operations"
    echo "Please install CUDA 12.0 or later from:"
    echo "  https://developer.nvidia.com/cuda-downloads"
fi

# Check for MPI
echo -e "\n${YELLOW}Checking for MPI...${NC}"
if command_exists mpirun; then
    echo -e "${GREEN}✓ MPI is installed${NC}"
    mpirun --version | head -n 1
else
    echo -e "${YELLOW}⚠ MPI not found${NC}"
    echo "Installing MPI..."
    
    case $OS in
        debian)
            sudo apt-get update
            sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
            ;;
        redhat)
            sudo yum install -y openmpi openmpi-devel
            ;;
        macos)
            if command_exists brew; then
                brew install open-mpi
            else
                echo "Please install Homebrew first: https://brew.sh"
                exit 1
            fi
            ;;
        *)
            echo "Please install MPI manually for your system"
            exit 1
            ;;
    esac
fi

# Check for Python
echo -e "\n${YELLOW}Checking for Python...${NC}"
if command_exists python3; then
    echo -e "${GREEN}✓ Python is installed${NC}"
    python3 --version
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.8 or later"
    exit 1
fi

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found${NC}"
    echo "Installing minimal dependencies..."
    pip3 install numpy matplotlib seaborn mpi4py
fi

# Create virtual environment (optional)
echo -e "\n${YELLOW}Do you want to create a Python virtual environment? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo "Activate with: source venv/bin/activate"
fi

# Set up environment variables
echo -e "\n${YELLOW}Setting up environment variables...${NC}"
cat > .env << EOF
# NVloom-Mojo Environment Variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export NVLOOM_HOME=$(pwd)
export PATH=\$NVLOOM_HOME/build:\$PATH
EOF

echo -e "${GREEN}✓ Environment file created (.env)${NC}"
echo "Source with: source .env"

# Build the project
echo -e "\n${YELLOW}Building NVloom-Mojo...${NC}"
if [ -f "Makefile" ]; then
    make clean
    make all
    echo -e "${GREEN}✓ Build complete${NC}"
else
    echo -e "${YELLOW}Using build script...${NC}"
    bash scripts/build.sh
fi

# Run tests
echo -e "\n${YELLOW}Running tests...${NC}"
if [ -f "build/examples" ]; then
    ./build/examples
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Test binary not found${NC}"
fi

# Final instructions
echo
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo
echo "To get started:"
echo "  1. Source the environment: source .env"
echo "  2. Run the CLI: ./build/nvloom_cli --help"
echo "  3. Run examples: ./build/examples"
echo
echo "Quick test:"
echo "  ./build/nvloom_cli -t bisect_device_to_device_write_sm"
echo
echo "Multi-GPU test (requires MPI):"
echo "  mpirun -np 4 ./build/nvloom_cli -s fabric-stress"
echo
echo "For more information, see README.md"
