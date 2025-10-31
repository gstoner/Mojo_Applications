# Contributing to NVloom-Mojo

Thank you for your interest in contributing to NVloom-Mojo! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Prioritize the community's best interests

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/nvloom-mojo.git
   cd nvloom-mojo
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/nvloom-mojo.git
   ```
4. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Create a detailed bug report including:
  - System information (OS, GPU model, CUDA version, Mojo version)
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Error messages and logs
  - Minimal reproducible example if possible

### Suggesting Features

- Check if the feature has already been suggested
- Create a feature request including:
  - Clear description of the feature
  - Use cases and benefits
  - Potential implementation approach
  - Any relevant examples or mockups

### Contributing Code

1. **Find an issue to work on** or create a new one
2. **Comment on the issue** to let others know you're working on it
3. **Write your code** following our coding standards
4. **Add tests** for your changes
5. **Update documentation** if needed
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Mojo 24.5 or later
- CUDA 12.0 or later
- Python 3.8+
- MPI implementation (OpenMPI or MPICH)
- Git

### Setup Instructions

1. Run the setup script:
   ```bash
   ./setup.sh
   ```

2. Or manually:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Build the project
   make clean && make all
   
   # Run tests
   make test
   ```

### Development Environment

We recommend using VS Code with the following extensions:
- Mojo Language Support
- Python
- C/C++ (for CUDA)
- GitLens
- EditorConfig

## Coding Standards

### Mojo Code Style

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Add type annotations where possible
- Document all public APIs with docstrings

Example:
```mojo
fn calculate_bandwidth(
    bytes_transferred: Int,
    time_seconds: Float64
) -> Float64:
    """Calculate bandwidth in GB/s.
    
    Args:
        bytes_transferred: Number of bytes transferred
        time_seconds: Time taken in seconds
    
    Returns:
        Bandwidth in gigabytes per second
    """
    let gb = Float64(bytes_transferred) / (1024 * 1024 * 1024)
    return gb / time_seconds
```

### Python Code Style

- Follow PEP 8
- Use Black for formatting
- Use type hints (Python 3.8+)

### Commit Messages

Follow the conventional commits specification:

```
type(scope): brief description

Longer explanation if needed. Wrap at 72 characters.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test suite
./scripts/run_tests.sh

# Run with coverage
make test-coverage

# Run benchmarks
make benchmark
```

### Writing Tests

- Add unit tests for new functions
- Add integration tests for new features
- Ensure tests are reproducible
- Mock external dependencies when appropriate

Example test:
```mojo
fn test_bandwidth_calculation():
    """Test bandwidth calculation function."""
    let bytes = 1024 * 1024 * 1024  # 1 GB
    let time = 1.0  # 1 second
    let expected = 1.0  # 1 GB/s
    
    let result = calculate_bandwidth(bytes, time)
    assert_almost_equal(result, expected, tolerance=0.001)
```

## Pull Request Process

1. **Update your fork:**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Rebase your feature branch:**
   ```bash
   git checkout feature/your-feature
   git rebase main
   ```

3. **Push to your fork:**
   ```bash
   git push origin feature/your-feature
   ```

4. **Create a pull request** on GitHub

5. **PR Requirements:**
   - Clear title and description
   - Link to related issues
   - Pass all CI checks
   - Have at least one approval
   - No merge conflicts
   - Updated documentation if needed

6. **After approval:**
   - Squash commits if requested
   - Ensure branch is up to date
   - Merge will be performed by maintainers

## Issue Guidelines

### Good First Issues

Look for issues labeled `good first issue` if you're new to the project.

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `performance`: Performance-related issues
- `question`: Questions about the project

## Project Structure

```
nvloom-mojo/
├── src/
│   ├── core/        # Core library implementation
│   ├── kernels/     # CUDA kernel implementations
│   ├── cli/         # Command-line interface
│   └── viz/         # Visualization tools
├── examples/        # Example usage
├── tests/           # Test suites
├── scripts/         # Build and utility scripts
├── docs/            # Documentation
└── .github/         # GitHub-specific files
```

## Documentation

- Update README.md for user-facing changes
- Add inline documentation for all public APIs
- Update examples if API changes
- Add migration guides for breaking changes

## Getting Help

- Check the documentation
- Search existing issues
- Ask in discussions
- Contact maintainers

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- The AUTHORS file

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to NVloom-Mojo! Your efforts help make this project better for everyone.
