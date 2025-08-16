import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt




def plot_garbage_results(output_dir: str = "results/plots"):
    """
    Plot the results of the garbage evaluation from a CSV file.
    
    :param csv_file: Path to the CSV file containing the evaluation results.
    :param output_dir: Directory to save the plot.
    """
    garbanch_result_mapping = {"Baseline": "results/garbench_baseline_config.yaml_garbage.csv", "Max Area": "results/garbench_maxarea_config.yaml_garbage.csv"}
    max_area_results = pd.read_csv(garbanch_result_mapping["Max Area"])
    baseline_results = pd.read_csv(garbanch_result_mapping["Baseline"])
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(max_area_results['garbage_rate'], max_area_results['cache_hit_ratio'], marker='o', label='Max Area Eviction')
    plt.plot(baseline_results['garbage_rate'], baseline_results['cache_hit_ratio'], marker='o', label='Baseline Eviction (LRU)')
    plt.title('Cache Hit Ratio vs Garbage Rate')
    plt.xlabel('Garbage Rate')
    plt.ylabel('Cache Hit Ratio')
    plt.grid(False)
    plt.legend()

    
    # Save the plot
    plot_path = Path(output_dir) / 'garbage_evaluation_plot.pdf'
    plt.savefig(plot_path,format='pdf')
    plt.close()
    
    print(f"Garbench Plot saved to {plot_path}")


def plot_paws_results(output_dir: str = "results/plots"):
    """
    Plot the results of the PAWS evaluation from a CSV file.
    
    :param csv_file: Path to the CSV file containing the evaluation results.
    :param output_dir: Directory to save the plot.
    """
    paws_result_mapping = {"Baseline": "results/paws_baseline_conf.yaml_paws.csv", "Max Area": "results/paws_conf.yaml_paws.csv"}
    max_area_results = pd.read_csv(paws_result_mapping["Max Area"])
    baseline_results = pd.read_csv(paws_result_mapping["Baseline"])
    categories = ["true_evac_rate","false_evac_rate","true_keep_rate","false_keep_rate"]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    
    # Make bar plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, category in enumerate(categories):
        axes[i].bar(['Max Area', 'Baseline'], [max_area_results[category].mean(), baseline_results[category].mean()], color=['blue', 'orange'])
        title = category.replace('_', ' ').title()
        if "true" in category:
            title += "↑↑"
        else:
            title += "↓↓"
        axes[i].set_title(title)
        axes[i].set_ylabel('Rate')
        axes[i].set_ylim(0, 1)
        axes[i].grid(False)
    
    # plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / 'paws_evaluation_plot.pdf'
    plt.savefig(plot_path, format='pdf')
    plt.close()
    
    print(f"PAWS Plot saved to {plot_path}")
if __name__ == "__main__":
    plot_garbage_results()
    plot_paws_results()
    # You can specify the output directory if needed
    # plot_garbage_results(output_dir="results/plots")