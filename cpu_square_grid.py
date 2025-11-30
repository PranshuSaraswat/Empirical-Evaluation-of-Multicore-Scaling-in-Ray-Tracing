# cpu_square_grid.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import os

CSV = "results/results.csv"
OUTFILE = "plots/cpu_grid_color.png"

# Ensure this matches the thread values you used when running experiments
THREAD_ORDER = [1, 2, 4, 8, 16]
# If you want a custom resolution order set RES_ORDER = ["640x360", "1280x720", ...]
RES_ORDER = None  # auto-sort by resolution (pixels) if None

# Color scale for CPU_MAX (green→yellow→red)
cmap = LinearSegmentedColormap.from_list(
    "cpu_max_cmap",
    [(0, "green"), (0.5, "yellow"), (1, "red")]
)


def plot_cpu_grid(csv_path=CSV, outfile=OUTFILE):
    # load CSV
    df = pd.read_csv(csv_path)

    # coerce important columns to numeric (safe)
    df['threads'] = pd.to_numeric(df.get('threads', 0), errors='coerce').fillna(0).astype(int)
    df['width'] = pd.to_numeric(df.get('width', 0), errors='coerce').fillna(0).astype(int)
    df['height'] = pd.to_numeric(df.get('height', 0), errors='coerce').fillna(0).astype(int)
    df['cpu_avg'] = pd.to_numeric(df.get('cpu_avg', 0.0), errors='coerce').fillna(0.0)
    df['cpu_max'] = pd.to_numeric(df.get('cpu_max', 0.0), errors='coerce').fillna(0.0)

    # create resolution label
    df['res_label'] = df['width'].astype(str) + "x" + df['height'].astype(str)

    # group by res_label and threads, compute median only for numeric columns we care about
    numeric_cols = ['cpu_avg', 'cpu_max', 'width', 'height']
    grouped = df.groupby(['res_label', 'threads'])[numeric_cols].median().reset_index()

    # determine resolution order (sorted by pixel count) unless user set one
    if RES_ORDER is None:
        res_info = grouped[['res_label', 'width', 'height']].drop_duplicates().copy()
        res_info['pixels'] = res_info['width'] * res_info['height']
        res_info = res_info.sort_values('pixels')
        res_list = list(res_info['res_label'])
    else:
        res_list = RES_ORDER

    # pivot to matrices (rows: resolution, cols: threads)
    pivot_avg = grouped.pivot(index='res_label', columns='threads', values='cpu_avg').reindex(index=res_list, columns=THREAD_ORDER)
    pivot_max = grouped.pivot(index='res_label', columns='threads', values='cpu_max').reindex(index=res_list, columns=THREAD_ORDER)

    # fill missing combos with zeros (or NaN if you prefer)
    pivot_avg = pivot_avg.fillna(0.0)
    pivot_max = pivot_max.fillna(0.0)

    # plotting grid parameters
    nrows = len(pivot_avg.index)
    ncols = len(pivot_avg.columns)
    if nrows == 0 or ncols == 0:
        raise RuntimeError(f"No data to plot. Check {csv_path} and ensure it contains numeric cpu_avg/cpu_max and thread values.")

    cell_size = 1.0
    fig_w = max(6, ncols * 1.2)
    fig_h = max(4, nrows * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()  # so first resolution appears at the top

    for i, res in enumerate(pivot_avg.index):
        for j, t in enumerate(pivot_avg.columns):
            avg_val = float(pivot_avg.loc[res, t])
            max_val = float(pivot_max.loc[res, t])

            # Outer square darkness based on CPU_AVG (0->white, 100->black)
            darkness = 1 - np.clip(avg_val / 100.0, 0.0, 1.0)
            outer_color = (darkness, darkness, darkness)

            # Inner square color for CPU_MAX using cmap (0->green, 1->red)
            max_norm = np.clip(max_val / 100.0, 0.0, 1.0)
            inner_color = cmap(max_norm)

            # Draw outer square
            rect = Rectangle((j, i), cell_size, cell_size, facecolor=outer_color, edgecolor="black", linewidth=0.8)
            ax.add_patch(rect)

            # Draw inner filled square (smaller than cell)
            inner_size = cell_size * 0.7
            inset = (cell_size - inner_size) / 2.0
            inner_rect = Rectangle(
                (j + inset, i + inset),
                inner_size,
                inner_size,
                facecolor=inner_color,
                edgecolor="black",
                linewidth=0.6
            )
            ax.add_patch(inner_rect)

            # Text inside: AVG/MAX
            txt = f"{avg_val:.0f}/{max_val:.0f}"
            text_color = "white" if avg_val > 55 else "black"
            ax.text(j + 0.5 * cell_size, i + 0.5 * cell_size, txt, ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")

    # ticks and labels
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels([str(t) for t in pivot_avg.columns], fontsize=10)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(list(pivot_avg.index), fontsize=10)

    ax.set_xlabel("Threads")
    ax.set_ylabel("Resolution")
    ax.set_title("CPU Utilisation Grid\nOuter = CPU_AVG darkness, Inner = CPU_MAX color (green→red)\nText = AVG/MAX", fontsize=11)

    # legend for CPU_MAX colors
    from matplotlib import cm
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # right-side colorbar
    norm = plt.Normalize(vmin=0, vmax=100)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('CPU_MAX (%)')

    ax.set_aspect('equal')
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])  # make room for colorbar
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=220)
    plt.close(fig)
    print("Saved:", outfile)


if __name__ == "__main__":
    plot_cpu_grid()
