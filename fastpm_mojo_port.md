# FastPM to Mojo GPU Porting Guide

## Overview

This guide outlines the process of porting FastPM, a cosmological N-body particle mesh solver, from C/MPI to Mojo for GPU acceleration. FastPM is currently a CPU-based parallel code that uses MPI and PFFT for distributed computing.

## Key Porting Considerations

### 1. Core Algorithm Components

FastPM's main computational components that need GPU acceleration:

- **Particle Mesh (PM) solver**: Density assignment and force interpolation
- **FFT operations**: Currently using PFFT, needs GPU-accelerated FFT
- **Particle updates**: Kick and drift operations on particle positions/velocities
- **Initial conditions**: 2LPT generation and white noise handling
- **Time stepping**: Modified kick-drift schemes

### 2. Memory Layout Transformation

**Current (C/MPI):**
```c
// Distributed across MPI ranks with domain decomposition
struct particle {
    double pos[3];     // Position in Mojo: Float64
    float vel[3];      // Velocity in km/s
    uint64_t id;       // Particle ID
};
```

**Target (Mojo GPU):**
```mojo
from memory import UnsafePointer
from gpu import DeviceContext, Buffer
from math import log2
import math

struct ParticleData:
    var positions: Buffer[DType.float64]  # [N, 3] array
    var velocities: Buffer[DType.float32] # [N, 3] array  
    var ids: Buffer[DType.uint64]         # [N] array
    
    fn __init__(inout self, num_particles: Int, ctx: DeviceContext):
        self.positions = Buffer[DType.float64].allocate(num_particles * 3, ctx)
        self.velocities = Buffer[DType.float32].allocate(num_particles * 3, ctx)
        self.ids = Buffer[DType.uint64].allocate(num_particles, ctx)
```

### 3. GPU Kernel Implementations

## Core GPU Kernels for FastPM

### 3.1 Density Assignment (NGP/CIC/TSC)

```mojo
from gpu import grid_dim, block_dim, threadIdx, blockIdx

@parameter
fn cic_density_kernel(
    positions: Buffer[DType.float64],
    density: Buffer[DType.float64],
    num_particles: Int,
    box_size: Float64,
    mesh_size: Int
):
    """Cloud-in-Cell density assignment kernel"""
    let tid = blockIdx().x * block_dim().x + threadIdx().x
    
    if tid >= num_particles:
        return
        
    # Get particle position
    let x = positions[tid * 3]
    let y = positions[tid * 3 + 1] 
    let z = positions[tid * 3 + 2]
    
    # Convert to grid coordinates
    let gx = x / box_size * Float64(mesh_size)
    let gy = y / box_size * Float64(mesh_size) 
    let gz = z / box_size * Float64(mesh_size)
    
    # Get grid indices and weights
    let i0 = Int(gx)
    let j0 = Int(gy)
    let k0 = Int(gz)
    
    let dx = gx - Float64(i0)
    let dy = gy - Float64(j0)
    let dz = gz - Float64(k0)
    
    # CIC weights for 8 surrounding cells
    let weights = [
        (1-dx)*(1-dy)*(1-dz), dx*(1-dy)*(1-dz),
        (1-dx)*dy*(1-dz),     dx*dy*(1-dz),
        (1-dx)*(1-dy)*dz,     dx*(1-dy)*dz,
        (1-dx)*dy*dz,         dx*dy*dz
    ]
    
    # Atomic addition to density grid (periodic boundary conditions)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                let ii = (i0 + i) % mesh_size
                let jj = (j0 + j) % mesh_size  
                let kk = (k0 + k) % mesh_size
                let idx = ii * mesh_size * mesh_size + jj * mesh_size + kk
                let weight_idx = i*4 + j*2 + k
                
                # Atomic add for thread safety
                atomic_add(density.unsafe_ptr() + idx, weights[weight_idx])
```

### 3.2 Force Interpolation

```mojo
@parameter
fn force_interpolation_kernel(
    positions: Buffer[DType.float64],
    forces: Buffer[DType.float64], 
    force_mesh: Buffer[DType.float64],
    num_particles: Int,
    box_size: Float64,
    mesh_size: Int
):
    """Interpolate forces from mesh to particles"""
    let tid = blockIdx().x * block_dim().x + threadIdx().x
    
    if tid >= num_particles:
        return
        
    let x = positions[tid * 3]
    let y = positions[tid * 3 + 1]
    let z = positions[tid * 3 + 2]
    
    # Convert to grid coordinates
    let gx = x / box_size * Float64(mesh_size)
    let gy = y / box_size * Float64(mesh_size)
    let gz = z / box_size * Float64(mesh_size)
    
    let i0 = Int(gx)
    let j0 = Int(gy) 
    let k0 = Int(gz)
    
    let dx = gx - Float64(i0)
    let dy = gy - Float64(j0)
    let dz = gz - Float64(k0)
    
    # Initialize force components
    var fx = Float64(0)
    var fy = Float64(0)  
    var fz = Float64(0)
    
    # Interpolate force from 8 surrounding mesh points
    for i in range(2):
        for j in range(2):
            for k in range(2):
                let ii = (i0 + i) % mesh_size
                let jj = (j0 + j) % mesh_size
                let kk = (k0 + k) % mesh_size
                
                let base_idx = (ii * mesh_size * mesh_size + jj * mesh_size + kk) * 3
                
                let weight = ((1-dx) if i==0 else dx) * \
                           ((1-dy) if j==0 else dy) * \
                           ((1-dz) if k==0 else dz)
                
                fx += weight * force_mesh[base_idx]
                fy += weight * force_mesh[base_idx + 1] 
                fz += weight * force_mesh[base_idx + 2]
    
    # Store interpolated forces
    forces[tid * 3] = fx
    forces[tid * 3 + 1] = fy
    forces[tid * 3 + 2] = fz
```

### 3.3 Particle Update (Kick-Drift)

```mojo
@parameter 
fn kick_drift_kernel(
    positions: Buffer[DType.float64],
    velocities: Buffer[DType.float32], 
    forces: Buffer[DType.float64],
    num_particles: Int,
    dt: Float64,
    kick_factor: Float64,
    drift_factor: Float64,
    box_size: Float64
):
    """Update particle positions and velocities"""
    let tid = blockIdx().x * block_dim().x + threadIdx().x
    
    if tid >= num_particles:
        return
        
    # Kick: update velocities
    velocities[tid * 3] += Float32(forces[tid * 3] * kick_factor)
    velocities[tid * 3 + 1] += Float32(forces[tid * 3 + 1] * kick_factor)
    velocities[tid * 3 + 2] += Float32(forces[tid * 3 + 2] * kick_factor)
    
    # Drift: update positions  
    positions[tid * 3] += Float64(velocities[tid * 3]) * drift_factor
    positions[tid * 3 + 1] += Float64(velocities[tid * 3 + 1]) * drift_factor
    positions[tid * 3 + 2] += Float64(velocities[tid * 3 + 2]) * drift_factor
    
    # Apply periodic boundary conditions
    positions[tid * 3] = fmod(positions[tid * 3], box_size)
    positions[tid * 3 + 1] = fmod(positions[tid * 3 + 1], box_size)  
    positions[tid * 3 + 2] = fmod(positions[tid * 3 + 2], box_size)
    
    # Handle negative coordinates
    if positions[tid * 3] < 0:
        positions[tid * 3] += box_size
    if positions[tid * 3 + 1] < 0:
        positions[tid * 3 + 1] += box_size
    if positions[tid * 3 + 2] < 0:
        positions[tid * 3 + 2] += box_size
```

## 4. FFT Implementation Strategy

FastPM heavily relies on FFT operations. For GPU acceleration:

```mojo
# Use cuFFT-compatible operations through Mojo's MLIR integration
from math.fft import fft3d, ifft3d
from gpu import DeviceBuffer

struct FastPMSolver:
    var mesh_size: Int
    var box_size: Float64
    var density_k: Buffer[DType.complex64]  # Fourier space density
    var force_k: Buffer[DType.complex64]    # Fourier space forces
    
    fn compute_forces(inout self, ctx: DeviceContext):
        # Forward FFT: density(x) -> density(k)
        fft3d(self.density_real, self.density_k, ctx)
        
        # Apply Green's function in Fourier space
        self.apply_greens_function(ctx)
        
        # Inverse FFT: force(k) -> force(x) for each component
        for component in range(3):
            ifft3d(self.force_k[component], self.force_real[component], ctx)
    
    @parameter
    fn apply_greens_function_kernel(
        density_k: Buffer[DType.complex64],
        force_k: Buffer[DType.complex64], 
        mesh_size: Int
    ):
        """Apply Green's function for gravitational force"""
        let tid = blockIdx().x * block_dim().x + threadIdx().x
        let total_size = mesh_size * mesh_size * mesh_size
        
        if tid >= total_size:
            return
            
        # Convert 1D index to 3D indices
        let kz = tid % mesh_size
        let ky = (tid // mesh_size) % mesh_size  
        let kx = tid // (mesh_size * mesh_size)
        
        # Calculate k-space coordinates (with proper aliasing)
        let kx_val = Float64(kx if kx <= mesh_size//2 else kx - mesh_size)
        let ky_val = Float64(ky if ky <= mesh_size//2 else ky - mesh_size)
        let kz_val = Float64(kz if kz <= mesh_size//2 else kz - mesh_size)
        
        let k_squared = kx_val*kx_val + ky_val*ky_val + kz_val*kz_val
        
        if k_squared > 0:
            let greens = -1.0 / k_squared  # Green's function for gravity
            
            # Compute force components: F_i = -i*k_i * Green's * density_k
            let density_val = density_k[tid]
            force_k[tid * 3] = Complex64(0, -kx_val * greens) * density_val
            force_k[tid * 3 + 1] = Complex64(0, -ky_val * greens) * density_val  
            force_k[tid * 3 + 2] = Complex64(0, -kz_val * greens) * density_val
```

## 5. Main Evolution Loop

```mojo
struct FastPMEvolution:
    var particles: ParticleData
    var solver: FastPMSolver
    var cosmology: CosmologyParams
    
    fn evolve_step(inout self, a_current: Float64, a_next: Float64, ctx: DeviceContext):
        """Single evolution step from a_current to a_next"""
        
        # Calculate time step factors
        let kick_factor = self.cosmology.kick_factor(a_current, a_next)
        let drift_factor = self.cosmology.drift_factor(a_current, a_next)
        
        # 1. Clear density mesh
        ctx.enqueue_function[clear_mesh_kernel](
            grid_dim=self.solver.mesh_size//64 + 1,
            block_dim=64,
            self.solver.density_mesh
        )
        
        # 2. Assign particles to mesh (density estimation)
        let num_blocks = (self.particles.num_particles + 255) // 256
        ctx.enqueue_function[cic_density_kernel](
            grid_dim=num_blocks,
            block_dim=256, 
            self.particles.positions,
            self.solver.density_mesh,
            self.particles.num_particles,
            self.solver.box_size,
            self.solver.mesh_size
        )
        
        # 3. Compute gravitational forces via FFT
        self.solver.compute_forces(ctx)
        
        # 4. Interpolate forces back to particles  
        ctx.enqueue_function[force_interpolation_kernel](
            grid_dim=num_blocks,
            block_dim=256,
            self.particles.positions,
            self.particles.forces,
            self.solver.force_mesh,
            self.particles.num_particles,
            self.solver.box_size,
            self.solver.mesh_size
        )
        
        # 5. Update particle positions and velocities
        ctx.enqueue_function[kick_drift_kernel](
            grid_dim=num_blocks, 
            block_dim=256,
            self.particles.positions,
            self.particles.velocities,
            self.particles.forces,
            self.particles.num_particles,
            drift_factor,
            kick_factor,
            self.solver.box_size
        )
        
        # Synchronize GPU operations
        ctx.synchronize()
```

## 6. Performance Optimizations

### Memory Coalescing
- Ensure particle data is stored in structure-of-arrays format for optimal GPU memory access
- Use shared memory for mesh operations where multiple threads access nearby cells

### Multi-GPU Scaling
```mojo
# For large simulations, implement domain decomposition across multiple GPUs
struct MultiGPUFastPM:
    var num_gpus: Int
    var gpu_contexts: List[DeviceContext]
    var local_particles: List[ParticleData]
    
    fn exchange_particles(inout self):
        """Exchange particles between GPU domains"""
        # Implement halo exchange between GPUs similar to MPI
        pass
```

### Tensor Core Utilization
For matrix operations in FFT or force calculations, leverage Mojo's Tensor Core support for maximum performance.

## 7. Integration with Existing FastPM Features

### Parameter File Support
Maintain compatibility with existing Lua parameter files by implementing a parser in Mojo.

### Output Format Compatibility  
Keep BigFile format compatibility for seamless integration with analysis tools like nbodykit.

### Initial Conditions
Port the 2LPT initial condition generator and maintain compatibility with existing input formats.

## Expected Performance Gains

Based on similar GPU ports of N-body codes:
- **10-100x speedup** for the particle-mesh operations
- **Significant memory bandwidth utilization** compared to CPU versions
- **Better scaling** for large particle counts due to massive GPU parallelism

The FlowPM project (TensorFlow implementation) showed ~10x speedups, and a well-optimized Mojo implementation should achieve similar or better performance due to lower-level GPU control.

## Migration Strategy

1. **Phase 1**: Port core PM solver (density assignment, FFT, force interpolation)
2. **Phase 2**: Implement time stepping and cosmological factors  
3. **Phase 3**: Add initial conditions and I/O compatibility
4. **Phase 4**: Optimize for multi-GPU and add advanced features
5. **Phase 5**: Validate against existing FastPM simulations

This approach leverages Mojo's Python-like syntax for rapid development while providing the performance benefits of low-level GPU programming.
