"""
Plot graphs from results CSV.
Produces:
 - plots/time_vs_threads_<WxH>.png
 - plots/speedup_vs_threads_<WxH>.png
 - plots/efficiency_vs_threads_<WxH>.png
 - plots/cpu_avg_vs_threads_<WxH>.png
 - plots/cpu_max_vs_threads_<WxH>.png
 - plots/time_vs_resolution_threads<threads>.png
 - plots/cpu_avg_vs_resolution_threads<threads>.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_for_resolution(df, width, height, outdir):
    sub = df[(df['width']==width) & (df['height']==height)].copy()
    # group by threads, take median time and median cpu stats
    g_time = sub.groupby('threads')['time_ms'].median().reset_index().sort_values('threads')
    g_cpu_avg = sub.groupby('threads')['cpu_avg'].median().reset_index().sort_values('threads')
    g_cpu_max = sub.groupby('threads')['cpu_max'].median().reset_index().sort_values('threads')

    threads = g_time['threads']
    times = g_time['time_ms']
    seq_time = float(g_time[g_time['threads']==1]['time_ms'].values[0]) if (g_time['threads']==1).any() else times.iloc[0]

    speedup = seq_time / times.replace(0, np.nan)
    efficiency = speedup / threads

    os.makedirs(outdir, exist_ok=True)

    # Time vs Threads
    plt.figure()
    plt.plot(threads, times, marker='o')
    plt.xlabel('Threads')
    plt.ylabel('Time (ms)')
    plt.title(f'Time vs Threads ({width}x{height})')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"time_vs_threads_{width}x{height}.png"), dpi=200)
    plt.close()

    # Speedup vs Threads
    plt.figure()
    plt.plot(threads, speedup, marker='o', label='Measured')
    plt.plot(threads, threads, linestyle='--', label='Ideal')
    plt.xlabel('Threads')
    plt.ylabel('Speedup')
    plt.title(f'Speedup vs Threads ({width}x{height})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"speedup_vs_threads_{width}x{height}.png"), dpi=200)
    plt.close()

    # Efficiency vs Threads
    plt.figure()
    plt.plot(threads, efficiency, marker='o')
    plt.xlabel('Threads')
    plt.ylabel('Efficiency')
    plt.title(f'Efficiency vs Threads ({width}x{height})')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"efficiency_vs_threads_{width}x{height}.png"), dpi=200)
    plt.close()

    # CPU Avg vs Threads
    plt.figure()
    plt.plot(g_cpu_avg['threads'], g_cpu_avg['cpu_avg'], marker='o')
    plt.xlabel('Threads')
    plt.ylabel('CPU Avg (%)')
    plt.title(f'CPU Avg vs Threads ({width}x{height})')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"cpu_avg_vs_threads_{width}x{height}.png"), dpi=200)
    plt.close()

    # CPU Max vs Threads
    plt.figure()
    plt.plot(g_cpu_max['threads'], g_cpu_max['cpu_max'], marker='o')
    plt.xlabel('Threads')
    plt.ylabel('CPU Max (%)')
    plt.title(f'CPU Max vs Threads ({width}x{height})')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"cpu_max_vs_threads_{width}x{height}.png"), dpi=200)
    plt.close()

def plot_time_vs_resolution(df, threads, outdir):
    sub = df[df['threads']==threads].copy()
    if sub.empty:
        print(f"No data for threads={threads} to plot time vs resolution.")
        return
    # group by resolution (pixels)
    sub['pixels'] = sub['width'] * sub['height']
    g = sub.groupby(['width','height','pixels'])['time_ms'].median().reset_index().sort_values('pixels')
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(g['pixels'], g['time_ms'], marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pixels (log)')
    plt.ylabel('Time (ms) (log)')
    plt.title(f'Time vs Resolution (threads={threads})')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"time_vs_resolution_threads{threads}.png"), dpi=200)
    plt.close()

def plot_cpu_vs_resolution(df, threads, outdir):
    sub = df[df['threads']==threads].copy()
    if sub.empty:
        print(f"No data for threads={threads} to plot CPU avg vs resolution.")
        return
    sub['pixels'] = sub['width'] * sub['height']
    g = sub.groupby(['width','height','pixels'])['cpu_avg'].median().reset_index().sort_values('pixels')
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(g['pixels'], g['cpu_avg'], marker='o')
    plt.xscale('log')
    plt.xlabel('Pixels (log)')
    plt.ylabel('CPU Avg (%)')
    plt.title(f'CPU Avg vs Resolution (threads={threads})')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"cpu_avg_vs_resolution_threads{threads}.png"), dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/results.csv')
    parser.add_argument('--outdir', default='plots')
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # coerce to numeric for key columns
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce').fillna(0)
    # If cpu columns missing, add zeros
    if 'cpu_avg' not in df.columns:
        df['cpu_avg'] = 0.0
    else:
        df['cpu_avg'] = pd.to_numeric(df['cpu_avg'], errors='coerce').fillna(0.0)
    if 'cpu_max' not in df.columns:
        df['cpu_max'] = 0.0
    else:
        df['cpu_max'] = pd.to_numeric(df['cpu_max'], errors='coerce').fillna(0.0)

    # pick unique resolutions
    resolutions = df[['width','height']].drop_duplicates().values.tolist()
    for w,h in resolutions:
        plot_for_resolution(df, int(w), int(h), args.outdir)

    plot_time_vs_resolution(df, args.threads, args.outdir)
    plot_cpu_vs_resolution(df, args.threads, args.outdir)

    print("Plots saved in", args.outdir)

if __name__ == '__main__':
    main()
