# Multi-stage Dockerfile for NVloom-Mojo
# Optimized for GPU workloads with CUDA and MPI support

# Stage 1: Build environment
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Mojo
RUN curl https://get.modular.com | sh && \
    modular install mojo
ENV PATH="/root/.modular/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Build NVloom
RUN make clean && make release

# Run tests
RUN make test

# Stage 2: Runtime environment
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir matplotlib seaborn numpy mpi4py && \
    rm /tmp/requirements.txt

# Copy built artifacts from builder
COPY --from=builder /app/build /opt/nvloom/bin
COPY --from=builder /app/scripts /opt/nvloom/scripts
COPY --from=builder /app/examples /opt/nvloom/examples

# Set working directory
WORKDIR /opt/nvloom

# Create wrapper script
RUN echo '#!/bin/bash\n\
export LD_LIBRARY_PATH=/opt/nvloom/bin:$LD_LIBRARY_PATH\n\
exec /opt/nvloom/bin/nvloom_cli "$@"' > /usr/local/bin/nvloom && \
    chmod +x /usr/local/bin/nvloom

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "cli" ]; then\n\
    shift\n\
    exec nvloom "$@"\n\
elif [ "$1" = "viz" ]; then\n\
    shift\n\
    exec /opt/nvloom/bin/plot_heatmaps "$@"\n\
elif [ "$1" = "examples" ]; then\n\
    exec /opt/nvloom/bin/examples\n\
elif [ "$1" = "bash" ]; then\n\
    exec /bin/bash\n\
else\n\
    exec nvloom "$@"\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

# Expose MPI port
EXPOSE 22

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]

# Labels
LABEL maintainer="NVloom-Mojo Contributors"
LABEL version="1.0.0"
LABEL description="High-performance MNNVL fabric testing tool written in Mojo"
