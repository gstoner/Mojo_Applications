# Makefile for NVloom-Mojo
# High-performance MNNVL fabric testing tool

MOJO := mojo
MOJO_FLAGS := -O3
CUDA_ARCH := sm_80
BUILD_DIR := build
SRC_DIR := src
EXAMPLES_DIR := examples
SCRIPTS_DIR := scripts

# Output binaries
NVLOOM_LIB := $(BUILD_DIR)/libnvloom.so
KERNELS_LIB := $(BUILD_DIR)/libkernels.so
CLI_BIN := $(BUILD_DIR)/nvloom_cli
VIZ_BIN := $(BUILD_DIR)/plot_heatmaps
EXAMPLES_BIN := $(BUILD_DIR)/examples

# Source files
CORE_SRC := $(SRC_DIR)/core/nvloom.mojo
KERNELS_SRC := $(SRC_DIR)/kernels/kernels.mojo
CLI_SRC := $(SRC_DIR)/cli/nvloom_cli.mojo
VIZ_SRC := $(SRC_DIR)/viz/plot_heatmaps.mojo
EXAMPLES_SRC := $(EXAMPLES_DIR)/examples.mojo

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

.PHONY: all clean install test help debug release setup

# Default target
all: setup $(NVLOOM_LIB) $(KERNELS_LIB) $(CLI_BIN) $(VIZ_BIN) $(EXAMPLES_BIN)
	@echo "$(GREEN)✓ Build complete!$(NC)"

# Setup build directory
setup:
	@mkdir -p $(BUILD_DIR)
	@echo "$(GREEN)✓ Build directory ready$(NC)"

# Build core library
$(NVLOOM_LIB): $(CORE_SRC)
	@echo "$(YELLOW)Building NVloom core library...$(NC)"
	@$(MOJO) build $(CORE_SRC) $(MOJO_FLAGS) -D CUDA_ARCH=$(CUDA_ARCH) -o $@ --shared
	@echo "$(GREEN)✓ Core library built$(NC)"

# Build kernels library
$(KERNELS_LIB): $(KERNELS_SRC)
	@echo "$(YELLOW)Building CUDA kernels...$(NC)"
	@$(MOJO) build $(KERNELS_SRC) $(MOJO_FLAGS) -D CUDA_ARCH=$(CUDA_ARCH) -o $@ --shared
	@echo "$(GREEN)✓ Kernels built$(NC)"

# Build CLI tool
$(CLI_BIN): $(CLI_SRC) $(NVLOOM_LIB) $(KERNELS_LIB)
	@echo "$(YELLOW)Building CLI tool...$(NC)"
	@$(MOJO) build $(CLI_SRC) $(MOJO_FLAGS) -o $@
	@echo "$(GREEN)✓ CLI tool built$(NC)"

# Build visualization tool
$(VIZ_BIN): $(VIZ_SRC)
	@echo "$(YELLOW)Building visualization tool...$(NC)"
	@$(MOJO) build $(VIZ_SRC) $(MOJO_FLAGS) -o $@
	@echo "$(GREEN)✓ Visualization tool built$(NC)"

# Build examples
$(EXAMPLES_BIN): $(EXAMPLES_SRC) $(NVLOOM_LIB)
	@echo "$(YELLOW)Building examples...$(NC)"
	@$(MOJO) build $(EXAMPLES_SRC) $(MOJO_FLAGS) -o $@
	@echo "$(GREEN)✓ Examples built$(NC)"

# Debug build
debug: MOJO_FLAGS := -O0 -g
debug: all

# Release build (default)
release: MOJO_FLAGS := -O3
release: all

# Install to system
install: all
	@echo "$(YELLOW)Installing NVloom...$(NC)"
	@sudo cp $(NVLOOM_LIB) $(KERNELS_LIB) /usr/local/lib/
	@sudo cp $(CLI_BIN) /usr/local/bin/nvloom_cli
	@sudo cp $(VIZ_BIN) /usr/local/bin/nvloom_viz
	@sudo ldconfig
	@echo "$(GREEN)✓ NVloom installed to /usr/local$(NC)"

# Run tests
test: all
	@echo "$(YELLOW)Running tests...$(NC)"
	@$(BUILD_DIR)/examples
	@echo "$(GREEN)✓ Tests passed$(NC)"

# Run quick benchmark
benchmark: $(CLI_BIN)
	@echo "$(YELLOW)Running quick benchmark...$(NC)"
	@$(CLI_BIN) -t bisect_device_to_device_write_sm -i 10
	@echo "$(GREEN)✓ Benchmark complete$(NC)"

# Clean build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)/*
	@rm -f *.so *.o
	@echo "$(GREEN)✓ Clean complete$(NC)"

# Show help
help:
	@echo "NVloom-Mojo Makefile"
	@echo "===================="
	@echo ""
	@echo "Targets:"
	@echo "  $(GREEN)all$(NC)        - Build everything (default)"
	@echo "  $(GREEN)debug$(NC)      - Build with debug symbols"
	@echo "  $(GREEN)release$(NC)    - Build with optimizations"
	@echo "  $(GREEN)install$(NC)    - Install to /usr/local"
	@echo "  $(GREEN)test$(NC)       - Run test suite"
	@echo "  $(GREEN)benchmark$(NC)  - Run quick benchmark"
	@echo "  $(GREEN)clean$(NC)      - Remove build artifacts"
	@echo "  $(GREEN)help$(NC)       - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH  - CUDA architecture (default: sm_80)"
	@echo "  MOJO_FLAGS - Mojo compiler flags"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build everything"
	@echo "  make debug              # Debug build"
	@echo "  make CUDA_ARCH=sm_90    # Build for H100"
	@echo "  make clean && make      # Clean rebuild"

# Specify CUDA architectures
cuda-kepler: CUDA_ARCH := sm_35
cuda-kepler: all

cuda-maxwell: CUDA_ARCH := sm_52
cuda-maxwell: all

cuda-pascal: CUDA_ARCH := sm_61
cuda-pascal: all

cuda-volta: CUDA_ARCH := sm_70
cuda-volta: all

cuda-ampere: CUDA_ARCH := sm_80
cuda-ampere: all

cuda-hopper: CUDA_ARCH := sm_90
cuda-hopper: all
