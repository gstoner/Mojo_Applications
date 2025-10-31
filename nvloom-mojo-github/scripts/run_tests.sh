#!/bin/bash

# Test runner script for NVloom-Mojo
# Runs various test configurations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BUILD_DIR="build"
RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/test_run_$TIMESTAMP.log"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Test function
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    log "${YELLOW}Running: $test_name${NC}"
    
    if eval "$test_cmd" >> "$LOG_FILE" 2>&1; then
        log "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        log "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}

# Header
log "${BLUE}================================${NC}"
log "${BLUE}NVloom-Mojo Test Suite${NC}"
log "${BLUE}================================${NC}"
log "Timestamp: $(date)"
log "Log file: $LOG_FILE"
log ""

# Check for required binaries
if [ ! -f "$BUILD_DIR/nvloom_cli" ]; then
    log "${RED}Error: nvloom_cli not found. Please run 'make' first.${NC}"
    exit 1
fi

if [ ! -f "$BUILD_DIR/examples" ]; then
    log "${RED}Error: examples not found. Please run 'make' first.${NC}"
    exit 1
fi

# Make binaries executable
chmod +x "$BUILD_DIR"/*

# Count available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
log "Available GPUs: $NUM_GPUS"
log ""

# Run unit tests
log "${BLUE}=== Unit Tests ===${NC}"
run_test "Example tests" "$BUILD_DIR/examples"
log ""

# Run CLI tests
log "${BLUE}=== CLI Tests ===${NC}"
run_test "CLI help" "$BUILD_DIR/nvloom_cli --help"
run_test "List testcases" "$BUILD_DIR/nvloom_cli -l"
log ""

# Single GPU tests (if GPU available)
if [ "$NUM_GPUS" -ge 1 ]; then
    log "${BLUE}=== Single GPU Tests ===${NC}"
    
    run_test "Bisect write SM" \
        "$BUILD_DIR/nvloom_cli -t bisect_device_to_device_write_sm -i 10"
    
    run_test "Bisect read SM" \
        "$BUILD_DIR/nvloom_cli -t bisect_device_to_device_read_sm -i 10"
    
    run_test "Small buffer test" \
        "$BUILD_DIR/nvloom_cli -t bisect_device_to_device_write_sm -b 1M -i 5"
    
    run_test "Large buffer test" \
        "$BUILD_DIR/nvloom_cli -t bisect_device_to_device_write_sm -b 1G -i 2"
    
    log ""
else
    log "${YELLOW}Skipping GPU tests (no GPUs available)${NC}"
    log ""
fi

# Multi-GPU tests (if multiple GPUs available)
if [ "$NUM_GPUS" -ge 2 ]; then
    log "${BLUE}=== Multi-GPU Tests ===${NC}"
    
    run_test "2-GPU pairwise" \
        "mpirun -np 2 $BUILD_DIR/nvloom_cli -t pairwise_device_to_device_write_sm -i 5"
    
    run_test "2-GPU fabric stress" \
        "mpirun -np 2 $BUILD_DIR/nvloom_cli -s fabric-stress -i 5"
    
    if [ "$NUM_GPUS" -ge 4 ]; then
        run_test "4-GPU bisect" \
            "mpirun -np 4 $BUILD_DIR/nvloom_cli -t bisect_device_to_device_write_sm -i 5"
        
        run_test "4-GPU gpu-to-rack" \
            "mpirun -np 4 $BUILD_DIR/nvloom_cli -s gpu-to-rack -i 5"
    fi
    
    if [ "$NUM_GPUS" -ge 8 ]; then
        run_test "8-GPU fabric stress" \
            "mpirun -np 8 $BUILD_DIR/nvloom_cli -s fabric-stress -i 3"
        
        run_test "8-GPU multicast" \
            "mpirun -np 8 $BUILD_DIR/nvloom_cli -t multicast_one_to_all -i 3"
    fi
    
    log ""
else
    log "${YELLOW}Skipping multi-GPU tests (need at least 2 GPUs)${NC}"
    log ""
fi

# Performance tests
log "${BLUE}=== Performance Tests ===${NC}"

if [ "$NUM_GPUS" -ge 1 ]; then
    run_test "Quick benchmark" \
        "timeout 10 $BUILD_DIR/nvloom_cli -t bisect_device_to_device_write_sm -i 100 || true"
    
    run_test "Duration test" \
        "timeout 5 $BUILD_DIR/nvloom_cli -t bisect_device_to_device_write_sm -d 3 || true"
else
    log "${YELLOW}Skipping performance tests (no GPUs available)${NC}"
fi
log ""

# Visualization tests
log "${BLUE}=== Visualization Tests ===${NC}"

if command -v python3 >/dev/null 2>&1; then
    run_test "Python dependencies" \
        "python3 -c 'import matplotlib, seaborn, numpy, mpi4py'"
    
    if [ -f "$BUILD_DIR/plot_heatmaps" ]; then
        run_test "Heatmap tool help" \
            "$BUILD_DIR/plot_heatmaps --help || true"
    fi
else
    log "${YELLOW}Python not available, skipping visualization tests${NC}"
fi
log ""

# Summary
log "${BLUE}================================${NC}"
log "${BLUE}Test Summary${NC}"
log "${BLUE}================================${NC}"

# Count results
TOTAL_TESTS=$(grep -c "Running:" "$LOG_FILE" || echo 0)
PASSED_TESTS=$(grep -c "✓" "$LOG_FILE" || echo 0)
FAILED_TESTS=$(grep -c "✗" "$LOG_FILE" || echo 0)
SKIPPED_TESTS=$(grep -c "Skipping" "$LOG_FILE" || echo 0)

log "Total tests: $TOTAL_TESTS"
log "Passed: ${GREEN}$PASSED_TESTS${NC}"
log "Failed: ${RED}$FAILED_TESTS${NC}"
log "Skipped: ${YELLOW}$SKIPPED_TESTS${NC}"
log ""

# Exit status
if [ "$FAILED_TESTS" -eq 0 ]; then
    log "${GREEN}All tests passed!${NC}"
    exit 0
else
    log "${RED}Some tests failed. Check $LOG_FILE for details.${NC}"
    exit 1
fi
