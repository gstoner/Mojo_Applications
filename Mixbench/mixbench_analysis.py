#!/usr/bin/env python3
"""
Analysis script for Mixbench-Mojo benchmark results
Parses CSV output and generates performance analysis and visualizations

Usage:
    python analyze_results.py benchmark_results.csv
    python analyze_results.py --plot --save benchmark_results.csv
"""

import argparse
import csv
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")

class BenchmarkData:
    """Container for benchmark data from a single precision type"""
    
    def __init__(self, name: str):
        self.name = name
        self.compute_iters: List[int] = []
        self.flops_per_byte: List[float] = []
        self.execution_times: List[float] = []
        self.performance: List[float] = []  # GFLOPS or GIOPS
        self.bandwidth: List[float] = []    # GB/s
    
    def add_data_point(self, compute_iters: int, flops_per_byte: float, 
                      exec_time: float, perf: float, bw: float):
        self.compute_iters.append(compute_iters)
        self.flops_per_byte.append(flops_per_byte)
        self.execution_times.append(exec_time)
        self.performance.append(perf)
        self.bandwidth.append(bw)
    
    def get_peak_performance(self) -> Tuple[float, int]:
        """Get peak performance and the compute iteration count where it occurs"""
        max_perf = max(self.performance)
        max_idx = self.performance.index(max_perf)
        return max_perf, self.compute_iters[max_idx]
    
    def get_peak_bandwidth(self) -> Tuple[float, int]:
        """Get peak bandwidth and the compute iteration count where it occurs"""
        max_bw = max(self.bandwidth)
        max_idx = self.bandwidth.index(max_bw)
        return max_bw, self.compute_iters[max_idx]
    
    def find_roofline_knee(self) -> Optional[Tuple[float, float]]:
        """Find the 'knee' point where compute becomes the bottleneck"""
        # Simple heuristic: find where performance growth rate drops significantly
        if len(self.performance) < 3:
            return None
            
        growth_rates = []
        for i in range(1, len(self.performance)):
            if self.flops_per_byte[i] > self.flops_per_byte[i-1]:  # Avoid division by zero
                rate = (self.performance[i] - self.performance[i-1]) / (self.flops_per_byte[i] - self.flops_per_byte[i-1])
                growth_rates.append((rate, i))
        
        if not growth_rates:
            return None
        
        # Find point where growth rate drops below 50% of maximum
        max_growth = max(rate for rate, _ in growth_rates)
        threshold = max_growth * 0.5
        
        for rate, idx in growth_rates:
            if rate < threshold:
                return self.flops_per_byte[idx], self.performance[idx]
        
        return None

class MixbenchAnalyzer:
    """Main analyzer for Mixbench results"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data: Dict[str, BenchmarkData] = {}
        self.parse_csv()
    
    def parse_csv(self):
        """Parse CSV file and extract benchmark data"""
        try:
            with open(self.csv_file, 'r') as f:
                # Find the CSV data section
                lines = f.readlines()
                csv_start = -1
                
                for i, line in enumerate(lines):
                    if "Compute iters, Flops/byte" in line:
                        csv_start = i + 1
                        break
                
                if csv_start == -1:
                    print(f"Error: Could not find CSV data section in {self.csv_file}")
                    return
                
                # Initialize data containers
                self.data["Single Precision"] = BenchmarkData("Single Precision (FP32)")
                self.data["Double Precision"] = BenchmarkData("Double Precision (FP64)")  
                self.data["Integer"] = BenchmarkData("Integer (INT32)")
                
                # Parse data lines
                for line in lines[csv_start:]:
                    line = line.strip()
                    if not line or line.startswith('-'):
                        continue
                        
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 13:  # Expected number of CSV columns
                        try:
                            compute_iters = int(parts[0])
                            
                            # Single precision data (columns 1-4)
                            sp_flops_per_byte = float(parts[1])
                            sp_exec_time = float(parts[2])
                            sp_gflops = float(parts[3])
                            sp_gbps = float(parts[4])
                            
                            # Double precision data (columns 5-8)
                            dp_flops_per_byte = float(parts[5])
                            dp_exec_time = float(parts[6])
                            dp_gflops = float(parts[7])
                            dp_gbps = float(parts[8])
                            
                            # Integer data (columns 9-12)
                            int_ops_per_byte = float(parts[9])
                            int_exec_time = float(parts[10])
                            int_giops = float(parts[11])
                            int_gbps = float(parts[12])
                            
                            self.data["Single Precision"].add_data_point(
                                compute_iters, sp_flops_per_byte, sp_exec_time, sp_gflops, sp_gbps)
                            self.data["Double Precision"].add_data_point(
                                compute_iters, dp_flops_per_byte, dp_exec_time, dp_gflops, dp_gbps)
                            self.data["Integer"].add_data_point(
                                compute_iters, int_ops_per_byte, int_exec_time, int_giops, int_gbps)
                                
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse line: {line[:50]}... (Error: {e})")
                            continue
                            
        except FileNotFoundError:
            print(f"Error: File {self.csv_file} not found")
            sys.exit(1)
    
    def print_summary(self):
        """Print summary analysis of the benchmark results"""
        print("="*80)
        print("MIXBENCH-MOJO ANALYSIS SUMMARY")
        print("="*80)
        print()
        
        for name, data in self.data.items():
            if not data.performance:
                continue
                
            print(f"{name} Results:")
            print("-" * (len(name) + 9))
            
            peak_perf, peak_perf_iters = data.get_peak_performance()
            peak_bw, peak_bw_iters = data.get_peak_bandwidth()
            
            print(f"  Peak Performance: {peak_perf:.2f} G{'FLOPS' if 'Precision' in name else 'IOPS'} "
                  f"(at {peak_perf_iters} compute iterations)")
            print(f"  Peak Bandwidth:   {peak_bw:.2f} GB/s (at {peak_bw_iters} compute iterations)")
            
            knee_point = data.find_roofline_knee()
            if knee_point:
                print(f"  Roofline Knee:    {knee_point[0]:.2f} ops/byte, {knee_point[1]:.2f} G{'FLOPS' if 'Precision' in name else 'IOPS'}")
            else:
                print("  Roofline Knee:    Not detected")
            
            # Memory vs compute bound analysis
            memory_bound_perf = np.mean(data.performance[:3]) if len(data.performance) >= 3 else data.performance[0]
            compute_bound_perf = np.mean(data.performance[-3:]) if len(data.performance) >= 3 else data.performance[-1]
            
            if compute_bound_perf / memory_bound_perf > 5:
                bottleneck = "Clearly compute-limited at high operational intensity"
            elif compute_bound_perf / memory_bound_perf > 2:
                bottleneck = "Moderately compute-limited at high operational intensity"
            else:
                bottleneck = "Bandwidth-limited across operational intensity range"
            
            print(f"  Bottleneck:       {bottleneck}")
            print()
    
    def generate_roofline_analysis(self):
        """Generate detailed roofline model analysis"""
        print("="*80)
        print("ROOFLINE MODEL ANALYSIS")
        print("="*80)
        print()
        
        for name, data in self.data.items():
            if not data.performance:
                continue
                
            print(f"{name} Roofline Analysis:")
            print("-" * (len(name) + 18))
            
            # Find the approximate roofline parameters
            peak_compute = max(data.performance)
            peak_memory = max(data.bandwidth)
            
            print(f"  Theoretical Peak Compute: {peak_compute:.2f} G{'FLOPS' if 'Precision' in name else 'IOPS'}")
            print(f"  Theoretical Peak Memory:  {peak_memory:.2f} GB/s")
            
            # Calculate computational intensity at ridge point
            # Ridge point: ops/byte = peak_compute / peak_memory
            ridge_intensity = peak_compute / peak_memory
            print(f"  Ridge Point:              {ridge_intensity:.3f} ops/byte")
            print(f"  Performance Regime:")
            
            low_intensity = data.flops_per_byte[0] if data.flops_per_byte else 0
            high_intensity = data.flops_per_byte[-1] if data.flops_per_byte else 0
            
            if low_intensity < ridge_intensity:
                print(f"    - Memory-bound region: {low_intensity:.3f} to {ridge_intensity:.3f} ops/byte")
            if high_intensity > ridge_intensity:
                print(f"    - Compute-bound region: {ridge_intensity:.3f} to {high_intensity:.3f} ops/byte")
                
            # Calculate efficiency
            actual_peak = max(data.performance)
            efficiency = (actual_peak / peak_compute) * 100 if peak_compute > 0 else 0
            print(f"  Compute Efficiency:       {efficiency:.1f}%")
            print()
    
    def plot_results(self, save_plots: bool = False, output_dir: str = "."):
        """Generate performance plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
            return
            
        # Set up plotting style
        style.use('seaborn-v0_8' if 'seaborn-v0_8' in style.available else 'default')
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Mixbench-Mojo Performance Analysis', fontsize=16, fontweight='bold')
        
        colors = {'Single Precision': '#1f77b4', 'Double Precision': '#ff7f0e', 'Integer': '#2ca02c'}
        
        # Plot 1: Roofline Model (Performance vs Operational Intensity)
        ax1.set_title('Roofline Model: Performance vs Operational Intensity', fontweight='bold')
        for name, data in self.data.items():
            if data.performance and data.flops_per_byte:
                ax1.loglog(data.flops_per_byte, data.performance, 'o-', 
                          color=colors[name], label=name, linewidth=2, markersize=5)
        
        ax1.set_xlabel('Operational Intensity (ops/byte)')
        ax1.set_ylabel('Performance (GFLOPS/GIOPS)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Performance vs Compute Iterations
        ax2.set_title('Performance vs Compute Iterations', fontweight='bold')
        for name, data in self.data.items():
            if data.performance and data.compute_iters:
                ax2.semilogx(data.compute_iters, data.performance, 'o-',
                           color=colors[name], label=name, linewidth=2, markersize=5)
        
        ax2.set_xlabel('Compute Iterations per Memory Access')
        ax2.set_ylabel('Performance (GFLOPS/GIOPS)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Memory Bandwidth Utilization
        ax3.set_title('Memory Bandwidth Utilization', fontweight='bold')
        for name, data in self.data.items():
            if data.bandwidth and data.compute_iters:
                ax3.plot(data.compute_iters, data.bandwidth, 'o-',
                        color=colors[name], label=name, linewidth=2, markersize=5)
        
        ax3.set_xlabel('Compute Iterations per Memory Access')
        ax3.set_ylabel('Bandwidth (GB/s)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Execution Time vs Compute Iterations  
        ax4.set_title('Execution Time vs Compute Iterations', fontweight='bold')
        for name, data in self.data.items():
            if data.execution_times and data.compute_iters:
                ax4.loglog(data.compute_iters, data.execution_times, 'o-',
                          color=colors[name], label=name, linewidth=2, markersize=5)
        
        ax4.set_xlabel('Compute Iterations per Memory Access')
        ax4.set_ylabel('Execution Time (ms)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{output_dir}/mixbench_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
            
            # Also save individual plots
            self._save_individual_plots(output_dir)
        else:
            plt.show()
    
    def _save_individual_plots(self, output_dir: str):
        """Save individual high-quality plots"""
        colors = {'Single Precision': '#1f77b4', 'Double Precision': '#ff7f0e', 'Integer': '#2ca02c'}
        
        # Roofline plot
        plt.figure(figsize=(10, 8))
        for name, data in self.data.items():
            if data.performance and data.flops_per_byte:
                plt.loglog(data.flops_per_byte, data.performance, 'o-', 
                          color=colors[name], label=name, linewidth=3, markersize=8)
        
        plt.xlabel('Operational Intensity (ops/byte)', fontsize=14)
        plt.ylabel('Performance (GFLOPS/GIOPS)', fontsize=14)
        plt.title('Roofline Model', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roofline_model.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plots saved in: {output_dir}/")
    
    def export_analysis_report(self, output_file: str = "mixbench_analysis_report.txt"):
        """Export detailed analysis report to text file"""
        with open(output_file, 'w') as f:
            f.write("MIXBENCH-MOJO DETAILED ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Source file: {self.csv_file}\n")
            f.write(f"Analysis date: {np.datetime64('now')}\n\n")
            
            for name, data in self.data.items():
                if not data.performance:
                    continue
                    
                f.write(f"{name} Detailed Analysis:\n")
                f.write("-" * (len(name) + 20) + "\n")
                
                peak_perf, peak_perf_iters = data.get_peak_performance()
                peak_bw, peak_bw_iters = data.get_peak_bandwidth()
                
                f.write(f"Peak Performance: {peak_perf:.2f} G{'FLOPS' if 'Precision' in name else 'IOPS'}\n")
                f.write(f"Peak Bandwidth: {peak_bw:.2f} GB/s\n")
                
                f.write("\nPerformance Data Points:\n")
                f.write("Compute_Iters, Ops/Byte, Exec_Time(ms), Performance, Bandwidth(GB/s)\n")
                
                for i in range(len(data.compute_iters)):
                    f.write(f"{data.compute_iters[i]:4d}, {data.flops_per_byte[i]:8.3f}, "
                           f"{data.execution_times[i]:10.2f}, {data.performance[i]:10.2f}, "
                           f"{data.bandwidth[i]:12.2f}\n")
                
                f.write("\n")
        
        print(f"Detailed analysis report saved as: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Mixbench-Mojo benchmark results')
    parser.add_argument('csv_file', help='CSV file containing benchmark results')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    parser.add_argument('--save', action='store_true', help='Save plots instead of displaying')
    parser.add_argument('--output-dir', default='.', help='Output directory for saved files')
    parser.add_argument('--report', action='store_true', help='Generate detailed analysis report')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary analysis')
    
    args = parser.parse_args()
    
    if not args.csv_file:
        print("Error: Please provide a CSV file to analyze")
        sys.exit(1)
    
    # Create analyzer and load data
    analyzer = MixbenchAnalyzer(args.csv_file)
    
    if not any(data.performance for data in analyzer.data.values()):
        print("Error: No valid benchmark data found in CSV file")
        sys.exit(1)
    
    # Print summary analysis
    analyzer.print_summary()
    
    if not args.summary_only:
        # Print roofline analysis
        analyzer.generate_roofline_analysis()
    
    # Generate plots if requested
    if args.plot:
        analyzer.plot_results(save_plots=args.save, output_dir=args.output_dir)
    
    # Generate detailed report if requested  
    if args.report:
        report_file = f"{args.output_dir}/mixbench_analysis_report.txt"
        analyzer.export_analysis_report(report_file)
    
    print("\nAnalysis complete!")
    if not args.plot and MATPLOTLIB_AVAILABLE:
        print("Tip: Use --plot to generate performance visualizations")
    if not args.report:
        print("Tip: Use --report to generate a detailed analysis report")

if __name__ == "__main__":
    main()
