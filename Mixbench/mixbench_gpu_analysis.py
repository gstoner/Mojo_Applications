#!/usr/bin/env python3
"""
GPU-specific analysis script for Mixbench-Mojo B100 results
Enhanced analysis for GPU benchmarks including FP8, Tensor Cores, and advanced metrics

Usage:
    python analyze_gpu_results.py benchmark_results.csv
    python analyze_gpu_results.py --gpu-metrics --plot --save results.csv
"""

import argparse
import csv
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")

class GPUBenchmarkData:
    """Enhanced container for GPU benchmark data"""
    
    def __init__(self, name: str, precision_type: str):
        self.name = name
        self.precision_type = precision_type  # "FP32", "FP64", "FP8", "TC"
        self.compute_iters: List[int] = []
        self.flops_per_byte: List[float] = []
        self.execution_times: List[float] = []
        self.performance: List[float] = []  # GFLOPS/GIOPS/TFLOPS
        self.bandwidth: List[float] = []    # GB/s
        
        # GPU-specific metrics
        self.gpu_utilization: List[float] = []
        self.memory_utilization: List[float] = []
        self.tensor_core_utilization: List[float] = []
        self.power_consumption: List[float] = []
        self.temperature: List[float] = []
    
    def add_data_point(self, compute_iters: int, flops_per_byte: float, 
                      exec_time: float, perf: float, bw: float, **kwargs):
        self.compute_iters.append(compute_iters)
        self.flops_per_byte.append(flops_per_byte)
        self.execution_times.append(exec_time)
        self.performance.append(perf)
        self.bandwidth.append(bw)
        
        # Optional GPU metrics
        self.gpu_utilization.append(kwargs.get('gpu_util', 0.0))
        self.memory_utilization.append(kwargs.get('mem_util', 0.0))
        self.tensor_core_utilization.append(kwargs.get('tc_util', 0.0))
        self.power_consumption.append(kwargs.get('power', 0.0))
        self.temperature.append(kwargs.get('temp', 0.0))
    
    def get_theoretical_peak(self) -> Tuple[float, str]:
        """Get theoretical peak performance based on precision type"""
        if self.precision_type == "FP8":
            return 5000000.0, "PFLOPS"  # B100 FP8 sparse peak
        elif self.precision_type == "TC":
            return 2000000.0, "TFLOPS"  # Tensor Core dense peak
        elif self.precision_type == "FP32":
            return 83000.0, "GFLOPS"    # B100 FP32 peak
        elif self.precision_type == "FP64":
            return 41500.0, "GFLOPS"    # B100 FP64 peak
        else:
            return max(self.performance) if self.performance else 0.0, "GFLOPS"
    
    def calculate_efficiency(self) -> float:
        """Calculate peak efficiency vs theoretical maximum"""
        if not self.performance:
            return 0.0
        theoretical_peak, _ = self.get_theoretical_peak()
        actual_peak = max(self.performance)
        return (actual_peak / theoretical_peak) * 100 if theoretical_peak > 0 else 0.0

class GPUMixbenchAnalyzer:
    """Enhanced analyzer for GPU Mixbench results"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data: Dict[str, GPUBenchmarkData] = {}
        self.gpu_info: Dict[str, str] = {}
        self.parse_gpu_csv()
    
    def parse_gpu_csv(self):
        """Parse GPU-specific CSV file format"""
        try:
            with open(self.csv_file, 'r') as f:
                lines = f.readlines()
                
                # Extract GPU information
                for line in lines:
                    if "Device:" in line:
                        self.gpu_info['device'] = line.split(":", 1)[1].strip()
                    elif "Total GPU memory:" in line:
                        self.gpu_info['memory'] = line.split(":", 1)[1].strip()
                    elif "Memory bandwidth:" in line:
                        self.gpu_info['bandwidth'] = line.split(":", 1)[1].strip()
                
                # Find CSV data section
                csv_start = -1
                for i, line in enumerate(lines):
                    if "Compute iters, Flops/byte" in line or "Matrix_size" in line:
                        csv_start = i + 1
                        break
                
                if csv_start == -1:
                    print(f"Error: Could not find GPU CSV data section in {self.csv_file}")
                    return
                
                # Initialize data containers for GPU precisions
                self.data["FP32"] = GPUBenchmarkData("Single Precision", "FP32")
                self.data["FP64"] = GPUBenchmarkData("Double Precision", "FP64")  
                self.data["FP8"] = GPUBenchmarkData("FP8 Precision", "FP8")
                self.data["Tensor Core"] = GPUBenchmarkData("Tensor Core GEMM", "TC")
                
                # Parse data lines
                for line in lines[csv_start:]:
                    line = line.strip()
                    if not line or line.startswith('-'):
                        continue
                        
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 17:  # Expected GPU CSV columns
                        try:
                            compute_iters = int(parts[0])
                            
                            # FP32 data (columns 1-4)
                            fp32_flops_per_byte = float(parts[1])
                            fp32_exec_time = float(parts[2])
                            fp32_gflops = float(parts[3])
                            fp32_gbps = float(parts[4])
                            
                            # FP64 data (columns 5-8)
                            fp64_flops_per_byte = float(parts[5])
                            fp64_exec_time = float(parts[6])
                            fp64_gflops = float(parts[7])
                            fp64_gbps = float(parts[8])
                            
                            # FP8 data (columns 9-12)
                            fp8_flops_per_byte = float(parts[9])
                            fp8_exec_time = float(parts[10])
                            fp8_gflops = float(parts[11])
                            fp8_gbps = float(parts[12])
                            
                            # Tensor Core data (columns 13-16)
                            tc_matrix_size = int(parts[13]) if parts[13] else 0
                            tc_exec_time = float(parts[14]) if parts[14] else 0.0
                            tc_gflops = float(parts[15]) if parts[15] else 0.0
                            tc_gbps = float(parts[16]) if parts[16] else 0.0
                            
                            self.data["FP32"].add_data_point(
                                compute_iters, fp32_flops_per_byte, fp32_exec_time, fp32_gflops, fp32_gbps)
                            self.data["FP64"].add_data_point(
                                compute_iters, fp64_flops_per_byte, fp64_exec_time, fp64_gflops, fp64_gbps)
                            self.data["FP8"].add_data_point(
                                compute_iters, fp8_flops_per_byte, fp8_exec_time, fp8_gflops, fp8_gbps)
                            
                            if tc_matrix_size > 0:
                                # Use matrix size as "compute iterations" for Tensor Core
                                self.data["Tensor Core"].add_data_point(
                                    tc_matrix_size, 0.0, tc_exec_time, tc_gflops, tc_gbps, tc_util=100.0)
                                
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse GPU line: {line[:50]}... (Error: {e})")
                            continue
                            
        except FileNotFoundError:
            print(f"Error: File {self.csv_file} not found")
            sys.exit(1)
    
    def print_gpu_summary(self):
        """Print GPU-specific summary analysis"""
        print("="*80)
        print("MIXBENCH-MOJO GPU ANALYSIS SUMMARY")
        print("="*80)
        print()
        
        # Print GPU information
        if self.gpu_info:
            print("GPU Information:")
            print("-" * 15)
            for key, value in self.gpu_info.items():
                print(f"  {key.title()}: {value}")
            print()
        
        for name, data in self.data.items():
            if not data.performance or max(data.performance) == 0:
                continue
                
            print(f"{name} ({data.precision_type}) Results:")
            print("-" * (len(name) + len(data.precision_type) + 12))
            
            peak_perf = max(data.performance)
            peak_idx = data.performance.index(peak_perf)
            peak_bw = max(data.bandwidth)
            
            theoretical_peak, unit = data.get_theoretical_peak()
            efficiency = data.calculate_efficiency()
            
            # Determine unit for display
            if data.precision_type == "FP8":
                display_unit = "PFLOPS" if peak_perf > 1000000 else "TFLOPS" if peak_perf > 1000 else "GFLOPS"
                display_perf = peak_perf / 1000000 if display_unit == "PFLOPS" else peak_perf / 1000 if display_unit == "TFLOPS" else peak_perf
            else:
                display_unit = "TFLOPS" if peak_perf > 1000 else "GFLOPS"
                display_perf = peak_perf / 1000 if display_unit == "TFLOPS" else peak_perf
            
            print(f"  Peak Performance: {display_perf:.2f} {display_unit}")
            print(f"  Peak Bandwidth:   {peak_bw:.2f} GB/s")
            print(f"  Theoretical Peak: {theoretical_peak:.0f} {unit}")
            print(f"  Efficiency:       {efficiency:.1f}%")
            
            if data.precision_type == "TC":
                print(f"  Optimal Matrix Size: {data.compute_iters[peak_idx]}")
            else:
                print(f"  Optimal Intensity: {data.compute_iters[peak_idx]} compute iterations")
            
            # Memory vs compute analysis
            if len(data.performance) >= 3:
                memory_bound_perf = np.mean(data.performance[:3])
                compute_bound_perf = np.mean(data.performance[-3:])
                
                if compute_bound_perf / memory_bound_perf > 10:
                    bottleneck = "Strongly compute-limited"
                elif compute_bound_perf / memory_bound_perf > 3:
                    bottleneck = "Compute-limited at high intensity"
                else:
                    bottleneck = "Memory bandwidth limited"
                
                print(f"  Bottleneck:       {bottleneck}")
            print()
    
    def generate_gpu_roofline_analysis(self):
        """Generate GPU-specific roofline analysis"""
        print("="*80)
        print("GPU ROOFLINE MODEL ANALYSIS")
        print("="*80)
        print()
        
        for name, data in self.data.items():
            if not data.performance or max(data.performance) == 0:
                continue
                
            print(f"{name} ({data.precision_type}) Roofline:")
            print("-" * (len(name) + len(data.precision_type) + 12))
            
            theoretical_peak, unit = data.get_theoretical_peak()
            peak_bandwidth = max(data.bandwidth) if data.bandwidth else 0
            
            print(f"  Compute Ceiling:  {theoretical_peak:.0f} {unit}")
            print(f"  Memory Ceiling:   {peak_bandwidth:.2f} GB/s")
            
            if theoretical_peak > 0 and peak_bandwidth > 0:
                ridge_point = theoretical_peak / peak_bandwidth
                print(f"  Ridge Point:      {ridge_point:.3f} ops/byte")
                
                # Classify performance regions
                if data.flops_per_byte:
                    low_intensity = min(data.flops_per_byte)
                    high_intensity = max(data.flops_per_byte)
                    
                    if low_intensity < ridge_point:
                        print(f"  Memory-bound:     {low_intensity:.3f} - {min(ridge_point, high_intensity):.3f} ops/byte")
                    if high_intensity > ridge_point:
                        print(f"  Compute-bound:    {ridge_point:.3f} - {high_intensity:.3f} ops/byte")
            print()
    
    def plot_gpu_results(self, save_plots: bool = False, output_dir: str = "."):
        """Generate GPU-specific performance plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib is required for plotting")
            return
            
        # Set up plotting style for GPU results
        style.use('dark_background' if any('B100' in str(v) for v in self.gpu_info.values()) else 'seaborn-v0_8')
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'GPU Mixbench Analysis - {self.gpu_info.get("device", "Unknown GPU")}', 
                     fontsize=18, fontweight='bold')
        
        # Define colors for different precisions
        colors = {'FP32': '#1f77b4', 'FP64': '#ff7f0e', 'FP8': '#2ca02c', 'Tensor Core': '#d62728'}
        
        # Plot 1: 3D Roofline Model
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.set_title('3D Performance Landscape', fontweight='bold')
        
        for name, data in self.data.items():
            if data.performance and data.flops_per_byte and data.execution_times:
                x = np.log10(np.array(data.flops_per_byte) + 1e-6)  # Operational intensity
                y = np.log10(np.array(data.execution_times) + 1e-6)  # Time
                z = np.array(data.performance)  # Performance
                
                ax1.scatter(x, y, z, c=colors[name], label=name, s=50, alpha=0.7)
        
        ax1.set_xlabel('Log10(Ops/Byte)')
        ax1.set_ylabel('Log10(Time ms)')
        ax1.set_zlabel('Performance')
        ax1.legend()
        
        # Plot 2: Traditional Roofline
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title('GPU Roofline Model', fontweight='bold')
        
        for name, data in self.data.items():
            if data.performance and data.flops_per_byte:
                ax2.loglog(data.flops_per_byte, data.performance, 'o-', 
                          color=colors[name], label=name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Operational Intensity (ops/byte)')
        ax2.set_ylabel('Performance (GFLOPS/TFLOPS)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Precision Comparison
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title('Peak Performance by Precision', fontweight='bold')
        
        precisions = []
        peak_performance = []
        efficiency_values = []
        
        for name, data in self.data.items():
            if data.performance and max(data.performance) > 0:
                precisions.append(data.precision_type)
                peak_performance.append(max(data.performance))
                efficiency_values.append(data.calculate_efficiency())
        
        bars = ax3.bar(precisions, peak_performance, color=[colors[p] for p in self.data.keys() if p in precisions])
        ax3.set_ylabel('Peak Performance')
        ax3.set_ylim(0, max(peak_performance) * 1.1 if peak_performance else 1)
        
        # Add efficiency labels on bars
        for bar, eff in zip(bars, efficiency_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Memory Bandwidth Utilization
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('Memory Bandwidth vs Compute Intensity', fontweight='bold')
        
        for name, data in self.data.items():
            if data.bandwidth and data.compute_iters:
                ax4.plot(data.compute_iters, data.bandwidth, 'o-',
                        color=colors[name], label=name, linewidth=2, markersize=5)
        
        ax4.set_xlabel('Compute Iterations / Matrix Size')
        ax4.set_ylabel('Bandwidth (GB/s)')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Execution Time Analysis
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title('Execution Time vs Problem Size', fontweight='bold')
        
        for name, data in self.data.items():
            if data.execution_times and data.compute_iters:
                ax5.loglog(data.compute_iters, data.execution_times, 'o-',
                          color=colors[name], label=name, linewidth=2, markersize=5)
        
        ax5.set_xlabel('Compute Iterations / Matrix Size')
        ax5.set_ylabel('Execution Time (ms)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Plot 6: Power Efficiency (if data available)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title('Performance per Precision Type', fontweight='bold')
        
        # Create radar chart for different metrics
        categories = ['Peak FLOPS', 'Peak BW', 'Efficiency', 'Intensity Range']
        
        # Normalize metrics for radar chart
        metrics_data = {}
        for name, data in self.data.items():
            if data.performance and max(data.performance) > 0:
                peak_flops = max(data.performance)
                peak_bw = max(data.bandwidth) if data.bandwidth else 0
                efficiency = data.calculate_efficiency()
                intensity_range = (max(data.flops_per_byte) - min(data.flops_per_byte)) if data.flops_per_byte else 0
                
                # Normalize to 0-100 scale
                metrics_data[name] = [
                    min(peak_flops / 1000, 100),  # Normalize FLOPS
                    min(peak_bw / 10, 100),       # Normalize bandwidth
                    efficiency,                    # Already 0-100
                    min(intensity_range * 10, 100)  # Normalize range
                ]
        
        # Simple bar chart instead of radar for easier implementation
        x_pos = np.arange(len(categories))
        bar_width = 0.15
        
        for i, (name, values) in enumerate(metrics_data.items()):
            ax6.bar(x_pos + i * bar_width, values, bar_width, 
                   color=colors[name], label=name, alpha=0.7)
        
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Normalized Score')
        ax6.set_xticks(x_pos + bar_width * (len(metrics_data) - 1) / 2)
        ax6.set_xticklabels(categories, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{output_dir}/gpu_mixbench_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"GPU analysis plot saved as: {filename}")
            
            # Save individual precision plots
            self._save_precision_plots(output_dir)
        else:
            plt.show()
    
    def _save_precision_plots(self, output_dir: str):
        """Save individual plots for each precision type"""
        colors = {'FP32': '#1f77b4', 'FP64': '#ff7f0e', 'FP8': '#2ca02c', 'Tensor Core': '#d62728'}
        
        for name, data in self.data.items():
            if not data.performance or max(data.performance) == 0:
                continue
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{name} ({data.precision_type}) Detailed Analysis', fontsize=16, fontweight='bold')
            
            # Performance vs Operational Intensity
            ax1.loglog(data.flops_per_byte, data.performance, 'o-', 
                      color=colors[name], linewidth=3, markersize=8)
            ax1.set_xlabel('Operational Intensity (ops/byte)')
            ax1.set_ylabel(f'Performance ({"TFLOPS" if max(data.performance) > 1000 else "GFLOPS"})')
            ax1.set_title('Roofline Performance')
            ax1.grid(True, alpha=0.3)
            
            # Bandwidth utilization
            ax2.plot(data.compute_iters, data.bandwidth, 'o-', 
                    color=colors[name], linewidth=3, markersize=8)
            ax2.set_xlabel('Problem Size')
            ax2.set_ylabel('Bandwidth (GB/s)')
            ax2.set_title('Memory Bandwidth')
            ax2.grid(True, alpha=0.3)
            
            # Execution time
            ax3.loglog(data.compute_iters, data.execution_times, 'o-', 
                      color=colors[name], linewidth=3, markersize=8)
            ax3.set_xlabel('Problem Size')
            ax3.set_ylabel('Execution Time (ms)')
            ax3.set_title('Timing Analysis')
            ax3.grid(True, alpha=0.3)
            
            # Efficiency vs problem size
            if len(data.performance) > 1:
                theoretical_peak, _ = data.get_theoretical_peak()
                efficiency = [(perf/theoretical_peak)*100 for perf in data.performance]
                ax4.plot(data.compute_iters, efficiency, 'o-', 
                        color=colors[name], linewidth=3, markersize=8)
                ax4.set_xlabel('Problem Size')
                ax4.set_ylabel('Efficiency (%)')
                ax4.set_title('Performance Efficiency')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 100)
            
            plt.tight_layout()
            filename = f"{output_dir}/{name.lower().replace(' ', '_')}_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual precision plots saved in: {output_dir}/")
    
    def export_gpu_json_report(self, output_file: str = "gpu_analysis_report.json"):
        """Export GPU analysis to JSON format"""
        report = {
            "gpu_info": self.gpu_info,
            "analysis_summary": {},
            "detailed_results": {}
        }
        
        for name, data in self.data.items():
            if not data.performance or max(data.performance) == 0:
                continue
                
            theoretical_peak, unit = data.get_theoretical_peak()
            efficiency = data.calculate_efficiency()
            
            summary = {
                "precision_type": data.precision_type,
                "peak_performance": max(data.performance),
                "peak_bandwidth": max(data.bandwidth) if data.bandwidth else 0,
                "theoretical_peak": theoretical_peak,
                "peak_unit": unit,
                "efficiency_percent": efficiency,
                "optimal_problem_size": data.compute_iters[data.performance.index(max(data.performance))],
                "performance_range": {
                    "min": min(data.performance),
                    "max": max(data.performance),
                    "mean": np.mean(data.performance),
                    "std": np.std(data.performance)
                }
            }
            
            detailed = {
                "compute_iterations": data.compute_iters,
                "operational_intensity": data.flops_per_byte,
                "execution_times_ms": data.execution_times,
                "performance_values": data.performance,
                "bandwidth_values": data.bandwidth
            }
            
            report["analysis_summary"][name] = summary
            report["detailed_results"][name] = detailed
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"GPU analysis report exported to: {output_file}")
    
    def compare_with_cpu(self, cpu_csv_file: str):
        """Compare GPU results with CPU baseline"""
        try:
            # This would load CPU results and compare
            print("GPU vs CPU Performance Comparison")
            print("=" * 40)
            print("Feature comparison:")
            print("  GPU Advantages:")
            print("    - Massive parallelism")
            print("    - High memory bandwidth")
            print("    - Specialized compute units (Tensor Cores)")
            print("    - Multiple precision support")
            print("")
            print("  CPU Advantages:")
            print("    - Lower latency")
            print("    - Better single-thread performance")
            print("    - More flexible control flow")
            print("    - Larger cache hierarchy")
            
        except Exception as e:
            print(f"Could not load CPU comparison data: {e}")
    
    def generate_optimization_recommendations(self):
        """Generate B100-specific optimization recommendations"""
        print("="*80)
        print("B100 GPU OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        print()
        
        for name, data in self.data.items():
            if not data.performance or max(data.performance) == 0:
                continue
                
            efficiency = data.calculate_efficiency()
            peak_idx = data.performance.index(max(data.performance))
            optimal_intensity = data.compute_iters[peak_idx]
            
            print(f"{name} ({data.precision_type}) Optimization:")
            print("-" * (len(name) + len(data.precision_type) + 15))
            
            if efficiency < 50:
                print("  ⚠️  Low efficiency detected! Recommendations:")
                if data.precision_type == "FP8":
                    print("    - Ensure sparsity is enabled for FP8 operations")
                    print("    - Use structured sparsity patterns (2:4)")
                    print("    - Optimize data layout for FP8 computations")
                elif data.precision_type == "TC":
                    print("    - Use optimal tile sizes (16x16 or 32x32)")
                    print("    - Ensure proper memory alignment")
                    print("    - Consider mixed-precision (FP16 input, FP32 accumulate)")
                else:
                    print("    - Increase computational intensity")
                    print("    - Optimize memory access patterns")
                    print("    - Consider using Tensor Cores when applicable")
            else:
                print("  ✅ Good efficiency achieved!")
                
            print(f"  📊 Optimal problem size: {optimal_intensity}")
            
            if data.precision_type == "FP8":
                print("  🎯 B100 FP8 specific tips:")
                print("    - Leverage Transformer Engine for automatic FP8")
                print("    - Use FP8 for forward pass, FP16/FP32 for backward pass")
                print("    - Enable FP8 in cuBLAS and cuDNN operations")
            elif data.precision_type == "TC":
                print("  🎯 Tensor Core optimization:")
                print("    - Use cuBLAS or cuDNN Tensor Core routines")
                print("    - Ensure matrix dimensions are multiples of 8/16")
                print("    - Consider async operations for better overlap")
            
            print()
        
        print("General B100 Optimization Tips:")
        print("  🚀 Memory:")
        print("    - Use HBM3e high bandwidth (8TB/s theoretical)")
        print("    - Minimize data transfers between host and device")
        print("    - Use unified memory when appropriate")
        print("")
        print("  🚀 Compute:")
        print("    - Target high occupancy (>75%)")
        print("    - Use async kernel launches")
        print("    - Consider multi-instance GPU (MIG) for workload isolation")
        print("")
        print("  🚀 Power/Thermal:")
        print("    - Monitor power consumption (700W TDP)")
        print("    - Use NVIDIA-SMI to track thermal throttling")
        print("    - Consider workload scheduling for thermal management")

def main():
    parser = argparse.ArgumentParser(description='Analyze GPU Mixbench-Mojo benchmark results')
    parser.add_argument('csv_file', help='CSV file containing GPU benchmark results')
    parser.add_argument('--plot', action='store_true', help='Generate GPU performance plots')
    parser.add_argument('--save', action='store_true', help='Save plots instead of displaying')
    parser.add_argument('--output-dir', default='.', help='Output directory for saved files')
    parser.add_argument('--json-report', action='store_true', help='Generate JSON analysis report')
    parser.add_argument('--compare-cpu', help='CPU CSV file for comparison')
    parser.add_argument('--gpu-metrics', action='store_true', help='Show detailed GPU metrics')
    parser.add_argument('--optimization-tips', action='store_true', help='Show optimization recommendations')
    parser.add_argument('--precision', choices=['FP32', 'FP64', 'FP8', 'TC', 'all'], 
                       default='all', help='Focus on specific precision type')
    
    args = parser.parse_args()
    
    if not args.csv_file:
        print("Error: Please provide a GPU CSV file to analyze")
        sys.exit(1)
    
    # Create analyzer and load data
    analyzer = GPUMixbenchAnalyzer(args.csv_file)
    
    if not any(max(data.performance) > 0 for data in analyzer.data.values() if data.performance):
        print("Error: No valid GPU benchmark data found in CSV file")
        sys.exit(1)
    
    # Print GPU summary analysis
    analyzer.print_gpu_summary()
    
    # Print roofline analysis
    analyzer.generate_gpu_roofline_analysis()
    
    # Show optimization recommendations
    if args.optimization_tips:
        analyzer.generate_optimization_recommendations()
    
    # Generate plots if requested
    if args.plot:
        analyzer.plot_gpu_results(save_plots=args.save, output_dir=args.output_dir)
    
    # Generate JSON report if requested  
    if args.json_report:
        report_file = f"{args.output_dir}/gpu_analysis_report.json"
        analyzer.export_gpu_json_report(report_file)
    
    # Compare with CPU if provided
    if args.compare_cpu:
        analyzer.compare_with_cpu(args.compare_cpu)
    
    print("\n🎉 GPU Analysis complete!")
    if not args.plot and MATPLOTLIB_AVAILABLE:
        print("💡 Tip: Use --plot to generate GPU performance visualizations")
    if not args.json_report:
        print("💡 Tip: Use --json-report to generate machine-readable analysis")
    if not args.optimization_tips:
        print("💡 Tip: Use --optimization-tips for B100-specific recommendations")

if __name__ == "__main__":
    main()
