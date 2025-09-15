# BabelStream Mojo Makefile
# Build system for the Mojo port of BabelStream

# Compiler and flags
MOJO = mojo
MOJO_FLAGS = -O3 --target=host

# Build directories
BUILD_DIR = build
SRC_DIR = src
TEST_DIR = tests
BENCH_DIR = benchmarks

# Source files
MAIN_SRC = $(SRC_DIR)/babelstream.mojo
GPU_SRC = $(SRC_DIR)/gpu_kernels.mojo
UTILS_SRC = $(SRC_DIR)/utils.mojo

# Target executables
CPU_TARGET = $(BUILD_DIR)/babelstream-cpu
GPU_TARGET = $(BUILD_DIR)/babelstream-gpu
TEST_TARGET = $(BUILD_DIR)/test-runner

# Default target
.PHONY: all
all: cpu

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# CPU-only build (default)
.PHONY: cpu
cpu: $(BUILD_DIR) $(CPU_TARGET)

$(CPU_TARGET): $(MAIN_SRC)
	$(MOJO) build $(MOJO_FLAGS) -D CPU_ONLY=1 -o $@ $<

# GPU build (when GPU support is available)
.PHONY: gpu
gpu: $(BUILD_DIR) $(GPU_TARGET)

$(GPU_TARGET): $(MAIN_SRC) $(GPU_SRC)
	$(MOJO) build $(MOJO_FLAGS) -D GPU_ENABLED=1 -o $@ $<

# Debug builds
.PHONY: debug
debug: $(BUILD_DIR)
	$(MOJO) build -O0 -g -D DEBUG=1 -o $(BUILD_DIR)/babelstream-debug $(MAIN_SRC)

# Profile build
.PHONY: profile
profile: $(BUILD_DIR)
	$(MOJO) build -O2 -g -D PROFILE=1 -o $(BUILD_DIR)/babelstream-profile $(MAIN_SRC)

# Test build
.PHONY: test
test: $(BUILD_DIR) $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_DIR)/test_kernels.mojo
	$(MOJO) build $(MOJO_FLAGS) -o $@ $<

# Benchmarking targets
.PHONY: bench
bench: cpu
	@echo "Running BabelStream CPU benchmark..."
	./$(CPU_TARGET)

.PHONY: bench-gpu
bench-gpu: gpu
	@echo "Running BabelStream GPU benchmark..."
	./$(GPU_TARGET)

.PHONY: bench-all
bench-all: cpu gpu
	@echo "Running comprehensive benchmark suite..."
	@echo "=== CPU Benchmark ==="
	./$(CPU_TARGET)
	@echo ""
	@echo "=== GPU Benchmark ==="
	./$(GPU_TARGET)

# Performance comparison
.PHONY: compare
compare: cpu
	@echo "Running performance comparison..."
	./$(CPU_TARGET) --csv > results/cpu_results.csv
	@if [ -f $(GPU_TARGET) ]; then \
		./$(GPU_TARGET) --csv > results/gpu_results.csv; \
		python3 scripts/compare_results.py results/cpu_results.csv results/gpu_results.csv; \
	fi

# Memory bandwidth specific benchmarks
.PHONY: bandwidth-test
bandwidth-test: cpu
	@echo "Testing memory bandwidth scaling..."
	@for size in 1048576 4194304 16777216 67108864 268435456; do \
		echo "Array size: $$size elements"; \
		./$(CPU_TARGET) --arraysize $$size --csv; \
		echo ""; \
	done

# Precision comparison
.PHONY: precision-test
precision-test: cpu
	@echo "Comparing single vs double precision..."
	@echo "=== Double Precision ==="
	./$(CPU_TARGET) --double
	@echo ""
	@echo "=== Single Precision ==="
	./$(CPU_TARGET) --float

# Kernel-specific benchmarks
.PHONY: triad-only
triad-only: cpu
	./$(CPU_TARGET) --triad-only

.PHONY: dot-only
dot-only: cpu
	./$(CPU_TARGET) --dot-only

# Validation
.PHONY: validate
validate: cpu
	@echo "Running validation tests..."
	./$(CPU_TARGET) --validate

# Clean targets
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: clean-results
clean-results:
	rm -rf results/*.csv results/*.json

.PHONY: distclean
distclean: clean clean-results

# Installation
PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin

.PHONY: install
install: cpu
	install -d $(BINDIR)
	install -m 755 $(CPU_TARGET) $(BINDIR)/babelstream-mojo
	@if [ -f $(GPU_TARGET) ]; then \
		install -m 755 $(GPU_TARGET) $(BINDIR)/babelstream-mojo-gpu; \
	fi

.PHONY: uninstall
uninstall:
	rm -f $(BINDIR)/babelstream-mojo
	rm -f $(BINDIR)/babelstream-mojo-gpu

# Development targets
.PHONY: format
format:
	find $(SRC_DIR) -name "*.mojo" -exec mojo format {} \;

.PHONY: lint
lint:
	find $(SRC_DIR) -name "*.mojo" -exec mojo lint {} \;

.PHONY: check
check: format lint test

# Documentation
.PHONY: docs
docs:
	mojo doc $(SRC_DIR)/*.mojo --output docs/

# Results directory
results:
	mkdir -p results

# Packaging
.PHONY: package
package: cpu
	mkdir -p package/babelstream-mojo
	cp $(CPU_TARGET) package/babelstream-mojo/
	cp README.md package/babelstream-mojo/
	cp LICENSE package/babelstream-mojo/
	@if [ -f $(GPU_TARGET) ]; then \
		cp $(GPU_TARGET) package/babelstream-mojo/; \
	fi
	tar -czf babelstream-mojo.tar.gz -C package babelstream-mojo

# Help target
.PHONY: help
help:
	@echo "BabelStream Mojo Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all            - Build CPU version (default)"
	@echo "  cpu            - Build CPU-only version"
	@echo "  gpu            - Build GPU version (when supported)"
	@echo "  debug          - Build debug version"
	@echo "  profile        - Build profiling version"
	@echo "  test           - Build and run tests"
	@echo ""
	@echo "Benchmarking:"
	@echo "  bench          - Run CPU benchmark"
	@echo "  bench-gpu      - Run GPU benchmark"
	@echo "  bench-all      - Run both CPU and GPU benchmarks"
	@echo "  bandwidth-test - Test different array sizes"
	@echo "  precision-test - Compare single vs double precision"
	@echo "  triad-only     - Run only triad kernel"
	@echo "  dot-only       - Run only dot product kernel"
	@echo "  validate       - Run validation tests"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          - Remove build directory"
	@echo "  format         - Format source code"
	@echo "  lint           - Run linter"
	@echo "  check          - Format, lint, and test"
	@echo "  docs           - Generate documentation"
	@echo "  package        - Create distribution package"
	@echo "  install        - Install to system (PREFIX=$(PREFIX))"
	@echo "  uninstall      - Remove from system"
	@echo ""
	@echo "Example usage:"
	@echo "  make cpu                    # Build CPU version"
	@echo "  make bench                  # Run benchmark"
	@echo "  make bench ARGS='--float'   # Run with single precision"
	@echo "  make install PREFIX=~/.local # Install to home directory"

# Include optional local configuration
-include Makefile.local