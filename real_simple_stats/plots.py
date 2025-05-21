import numpy as np
import matplotlib.pyplot as plt

def set_minimalist_style():
    """Apply a Tufte-inspired minimalist style to Matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "axes.grid": False,
        "grid.color": "white"
    })

def plot_norm_hist(data, mean, std, bins=30, show_pdf=True, show_lines=True, title=True):
    """Plot a histogram of data with optional normal curve and markers."""
    set_minimalist_style()

    _, bins_edges, _ = plt.hist(data, bins=bins, density=True, alpha=0.5, edgecolor='black')
    
    if show_pdf:
        x = np.linspace(min(bins_edges), max(bins_edges), 300)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        plt.plot(x, y, color='black', linewidth=1.5)

    if show_lines:
        plt.axvline(mean - 2*std, color='black', linestyle='--', linewidth=1)
        plt.axvline(mean + 2*std, color='black', linestyle='--', linewidth=1)

    if title:
        plt.title(f"Normal Distribution (μ = {mean:.2f}, σ = {std:.2f})")
    
    plt.savefig("norm_hist.png", bbox_inches="tight", dpi=300)
    plt.show()

def plot_box(data, showfliers=False):
    """Plot a horizontal boxplot without outliers."""
    set_minimalist_style()
    
    fig, ax = plt.subplots()
    ax.boxplot(data, vert=False, showfliers=showfliers, patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               medianprops=dict(color='black'))

    ax.set_title("Boxplot")
    
    plt.savefig("boxplot.png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_observed_vs_expected(observed, expected, title="Observed vs Expected"):
    """
    Minimalist bar plot comparing observed and expected frequencies.

    Args:
        observed: List of observed counts.
        expected: List of expected counts.
        title: Plot title.
    """
    set_minimalist_style()
    
    x = range(len(observed))
    width = 0.4
    plt.bar([i - width/2 for i in x], observed, width=width, label="Observed", color='black', alpha=0.7)
    plt.bar([i + width/2 for i in x], expected, width=width, label="Expected", color='gray', alpha=0.5)

    plt.xticks(x)
    plt.title(title)
    plt.legend()
    plt.savefig("observed_vs_expected.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Example usage
    import numpy as np

    mu, sigma = 50, 10
    data = np.random.normal(mu, sigma, 1000)

    plot_norm_hist(data, mu, sigma)
    plot_box(data)
