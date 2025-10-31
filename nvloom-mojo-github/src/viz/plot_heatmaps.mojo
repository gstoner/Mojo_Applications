"""
Heatmap Visualization for NVloom Results
Generates bandwidth heatmaps from test results
"""

from python import Python
from collections import DynamicVector

struct HeatmapGenerator:
    """Generate heatmap visualizations from NVloom results"""
    var output_path: String
    var plot_size: Int
    var title_fontsize: Int
    var legend_fontsize: Int
    var data_fontsize_scaling: Float64
    var heatmap_lower_limit: Float64
    var heatmap_upper_limit: Float64
    
    fn __init__(inout self):
        self.output_path = "."
        self.plot_size = 32
        self.title_fontsize = 32
        self.legend_fontsize = 24
        self.data_fontsize_scaling = 1.0
        self.heatmap_lower_limit = 0.0
        self.heatmap_upper_limit = -1.0  # -1 means auto
    
    fn parse_args(inout self, args: DynamicVector[String]):
        """Parse command line arguments"""
        var i = 1
        while i < len(args):
            let arg = args[i]
            
            if arg == "-p" or arg == "--path":
                i += 1
                if i < len(args):
                    self.output_path = args[i]
            
            elif arg == "-l" or arg == "--heatmap_lower_limit":
                i += 1
                if i < len(args):
                    self.heatmap_lower_limit = Float64(args[i])
            
            elif arg == "-u" or arg == "--heatmap_upper_limit":
                i += 1
                if i < len(args):
                    self.heatmap_upper_limit = Float64(args[i])
            
            elif arg == "--title_fontsize":
                i += 1
                if i < len(args):
                    self.title_fontsize = int(args[i])
            
            elif arg == "--legend_fontsize":
                i += 1
                if i < len(args):
                    self.legend_fontsize = int(args[i])
            
            elif arg == "--data_fontsize_scaling_factor":
                i += 1
                if i < len(args):
                    self.data_fontsize_scaling = Float64(args[i])
            
            elif arg == "--plot_size":
                i += 1
                if i < len(args):
                    self.plot_size = int(args[i])
            
            i += 1
    
    fn generate_heatmap(self, results: TestResult) raises:
        """Generate heatmap for test results"""
        let plt = Python.import_module("matplotlib.pyplot")
        let np = Python.import_module("numpy")
        let sns = Python.import_module("seaborn")
        
        # Extract GPU count from results
        var max_gpu = 0
        for i in range(len(results.measurements)):
            let m = results.measurements[i]
            if m.source_gpu > max_gpu:
                max_gpu = m.source_gpu
            if m.dest_gpu > max_gpu:
                max_gpu = m.dest_gpu
        
        let num_gpus = max_gpu + 1
        
        # Create bandwidth matrix
        let matrix = np.zeros((num_gpus, num_gpus))
        
        for i in range(len(results.measurements)):
            let m = results.measurements[i]
            matrix[m.source_gpu, m.dest_gpu] = m.bandwidth_gbps
        
        # Create figure
        let fig = plt.figure(figsize=(self.plot_size, self.plot_size))
        let ax = fig.add_subplot(111)
        
        # Set color limits
        var vmin = self.heatmap_lower_limit
        var vmax = self.heatmap_upper_limit
        
        if vmax < 0:  # Auto mode
            vmax = np.max(matrix)
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={"label": "Bandwidth (GB/s)"},
            annot_kws={"fontsize": int(self.legend_fontsize * self.data_fontsize_scaling)},
            ax=ax
        )
        
        # Set labels
        ax.set_xlabel("Destination GPU", fontsize=self.legend_fontsize)
        ax.set_ylabel("Source GPU", fontsize=self.legend_fontsize)
        ax.set_title(results.test_name, fontsize=self.title_fontsize)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        let filename = self.output_path + "/" + results.test_name + "_heatmap.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()
        
        print("Generated heatmap:", filename)
    
    fn generate_rack_heatmap(self, results: TestResult, gpus_per_rack: Int) raises:
        """Generate rack-level heatmap"""
        let plt = Python.import_module("matplotlib.pyplot")
        let np = Python.import_module("numpy")
        let sns = Python.import_module("seaborn")
        
        # Calculate number of racks
        var max_gpu = 0
        for i in range(len(results.measurements)):
            if results.measurements[i].source_gpu > max_gpu:
                max_gpu = results.measurements[i].source_gpu
        
        let num_racks = (max_gpu + gpus_per_rack) // gpus_per_rack
        
        # Create rack bandwidth matrix
        let matrix = np.zeros((num_racks, num_racks))
        let counts = np.zeros((num_racks, num_racks))
        
        for i in range(len(results.measurements)):
            let m = results.measurements[i]
            let src_rack = m.source_gpu // gpus_per_rack
            let dst_rack = m.dest_gpu // gpus_per_rack
            
            matrix[src_rack, dst_rack] += m.bandwidth_gbps
            counts[src_rack, dst_rack] += 1
        
        # Average the bandwidths
        for i in range(num_racks):
            for j in range(num_racks):
                if counts[i, j] > 0:
                    matrix[i, j] /= counts[i, j]
        
        # Create figure
        let fig = plt.figure(figsize=(self.plot_size // 2, self.plot_size // 2))
        let ax = fig.add_subplot(111)
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=self.heatmap_lower_limit,
            vmax=self.heatmap_upper_limit if self.heatmap_upper_limit > 0 else np.max(matrix),
            square=True,
            cbar_kws={"label": "Avg Bandwidth (GB/s)"},
            ax=ax
        )
        
        # Set labels
        ax.set_xlabel("Destination Rack", fontsize=self.legend_fontsize)
        ax.set_ylabel("Source Rack", fontsize=self.legend_fontsize)
        ax.set_title("Rack-to-Rack Bandwidth", fontsize=self.title_fontsize)
        
        # Set tick labels
        rack_labels = [f"Rack {i}" for i in range(num_racks)]
        ax.set_xticklabels(rack_labels)
        ax.set_yticklabels(rack_labels)
        
        plt.tight_layout()
        
        # Save figure
        let filename = self.output_path + "/rack_to_rack_heatmap.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()
        
        print("Generated rack heatmap:", filename)
    
    fn generate_summary_plots(self, all_results: DynamicVector[TestResult]) raises:
        """Generate summary plots for all test results"""
        let plt = Python.import_module("matplotlib.pyplot")
        let np = Python.import_module("numpy")
        
        # Create figure with subplots
        let num_tests = len(all_results)
        let cols = min(3, num_tests)
        let rows = (num_tests + cols - 1) // cols
        
        let fig = plt.figure(figsize=(cols * 8, rows * 6))
        
        for i in range(num_tests):
            let result = all_results[i]
            let ax = fig.add_subplot(rows, cols, i + 1)
            
            # Extract bandwidth values
            var bandwidths = DynamicVector[Float64]()
            for j in range(len(result.measurements)):
                bandwidths.append(result.measurements[j].bandwidth_gbps)
            
            # Convert to numpy array
            let bw_array = np.array([bw for bw in bandwidths])
            
            # Create histogram
            ax.hist(bw_array, bins=30, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Bandwidth (GB/s)")
            ax.set_ylabel("Frequency")
            ax.set_title(result.test_name)
            ax.axvline(result.summary_stats.mean_bandwidth, color="red", 
                      linestyle="--", label=f"Mean: {result.summary_stats.mean_bandwidth:.1f}")
            ax.axvline(result.summary_stats.median_bandwidth, color="green",
                      linestyle="--", label=f"Median: {result.summary_stats.median_bandwidth:.1f}")
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        let filename = self.output_path + "/summary_histograms.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()
        
        print("Generated summary plots:", filename)

fn parse_nvloom_output(input_file: String) -> DynamicVector[TestResult]:
    """Parse NVloom output file to extract results"""
    var results = DynamicVector[TestResult]()
    
    # This would parse the actual output format
    # For now, returning empty results
    
    return results

fn main() raises:
    """Main entry point for heatmap generation"""
    import sys
    
    print("NVloom Heatmap Generator")
    print("========================")
    
    # Parse arguments
    var args = DynamicVector[String]()
    for i in range(len(sys.argv)):
        args.append(sys.argv[i])
    
    var generator = HeatmapGenerator()
    generator.parse_args(args)
    
    # Read input from stdin or file
    var input_file = ""
    if len(args) > 1 and not args[-1].startswith("-"):
        input_file = args[-1]
    
    # Parse results
    let results = parse_nvloom_output(input_file)
    
    if len(results) == 0:
        print("No results found in input")
        return
    
    # Generate heatmaps
    for i in range(len(results)):
        generator.generate_heatmap(results[i])
    
    # Generate summary plots
    generator.generate_summary_plots(results)
    
    print("\nHeatmap generation complete!")
