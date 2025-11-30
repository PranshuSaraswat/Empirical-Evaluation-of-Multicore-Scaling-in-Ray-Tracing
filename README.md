# Empirical Evaluation of Multicore Scaling in Ray Tracing

A Python-based study analyzing how ray tracing performance scales with multiple CPU cores. This project implements a tile-based parallel ray tracer and benchmarks its performance across different thread counts and image resolutions.

## 🎯 Overview

This project investigates the relationship between CPU parallelism and rendering performance using a custom ray tracer. Key metrics analyzed include:
- **Execution Time** - Wall-clock time for rendering
- **Speedup** - Performance improvement vs sequential execution
- **Parallel Efficiency** - How effectively additional cores are utilized
- **CPU Utilization** - Average and maximum CPU usage during renders

## ✨ Features

- **Ray Tracer** with Phong shading, reflections, and shadow casting
- **Tile-based parallelization** using Python's `multiprocessing` module
- **Automated benchmarking** across multiple configurations
- **Comprehensive visualization** of performance metrics
- **IEEE-format research paper** with analysis and findings

## 📁 Project Structure

```
├── raytracer.py          # Main ray tracer (sequential & parallel modes)
├── run_experiments.py    # Batch runner for benchmarking
├── plot_results.py       # Graph generation for all metrics
├── cpu_square_grid.py    # CPU utilization grid visualization
├── compare_images.py     # Image comparison utility
├── requirements.txt      # Python dependencies
├── paper.tex             # IEEE conference paper
├── howto.txt             # Detailed usage instructions
├── scenes/               # Scene definition files (JSON)
│   ├── scene1.json       # Single sphere
│   ├── scene2.json       # Multiple spheres with 2 lights
│   ├── scene3.json       # Reflective spheres
│   └── scene4.json       # Complex scene (12 spheres)
├── results/              # Benchmark data and rendered images
│   └── results.csv       # Raw experimental data
└── plots/                # Generated performance graphs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run a Single Render

**Sequential (1 thread):**
```bash
python raytracer.py --scene scenes/scene1.json --width 800 --height 600 --threads 1 --outfile results/seq_test.png
```

**Parallel (8 threads):**
```bash
python raytracer.py --scene scenes/scene2.json --width 1920 --height 1080 --threads 8 --outfile results/par_test.png
```

### 3. Run Full Benchmark Suite

```bash
python run_experiments.py --scene scenes/scene2.json --out results/results.csv
```

This tests:
- Resolutions: 640×360, 1280×720, 1920×1080, 3840×2160
- Thread counts: 1, 2, 4, 8, 16
- 3 repeats per configuration

### 4. Generate Plots

```bash
python plot_results.py --csv results/results.csv --outdir plots --threads 16
```

### 5. Compare Images (Correctness Check)

```bash
python compare_images.py results/seq_test.png results/par_test.png
```

## 📊 Key Findings

| Threads | Speedup (1920×1080) | Efficiency |
|---------|---------------------|------------|
| 1       | 1.00×               | 100%       |
| 2       | ~1.3×               | ~65%       |
| 4       | ~3.5×               | ~88%       |
| 8       | ~5.7×               | ~71%       |
| 16      | ~6.3×               | ~39%       |

- **Optimal scaling** observed up to 8 threads
- **Diminishing returns** beyond 8 threads due to overhead
- **Higher resolutions** scale better with more threads
- **CPU utilization** increases but doesn't always correlate with speedup

## 🔧 Command Line Options

### raytracer.py
| Option | Description | Default |
|--------|-------------|---------|
| `--scene` | Path to scene JSON file | Required |
| `--width` | Image width in pixels | 800 |
| `--height` | Image height in pixels | 600 |
| `--threads` | Number of worker processes | 1 |
| `--tile` | Tile size for parallel rendering | 16 |
| `--max_depth` | Maximum reflection depth | 2 |
| `--outfile` | Output image path | out.png |
| `--mode` | `auto`, `sequential`, or `parallel` | auto |

### run_experiments.py
| Option | Description | Default |
|--------|-------------|---------|
| `--scene` | Path to scene JSON file | Required |
| `--out` | Output CSV file path | results/results.csv |
| `--threads` | Comma-separated thread counts | 1,2,4,8,16 |
| `--res` | Comma-separated resolutions | 640x360,1280x720,... |
| `--tile` | Tile size | 16 |
| `--repeats` | Number of repeats per config | 3 |

## 📈 Generated Plots

The `plots/` directory contains:
- `time_vs_threads_<resolution>.png` - Render time scaling
- `speedup_vs_threads_<resolution>.png` - Speedup curves with ideal reference
- `efficiency_vs_threads_<resolution>.png` - Parallel efficiency
- `cpu_avg_vs_threads_<resolution>.png` - Average CPU utilization
- `cpu_max_vs_threads_<resolution>.png` - Maximum CPU utilization
- `time_vs_resolution_threads<n>.png` - Resolution scaling
- `cpu_grid_color.png` - Combined CPU utilization heatmap

## 🧪 Test Environment

- **CPU**: AMD Ryzen 9 5900HX (8 cores, 16 threads)
- **OS**: Windows 11
- **Python**: 3.x with multiprocessing
- **Key Libraries**: numpy, pillow, matplotlib, pandas, psutil

## 📚 Dependencies

```
numpy
pillow
matplotlib
pandas
psutil
```

## 📄 Research Paper

The `paper.tex` file contains a complete IEEE-format conference paper documenting:
- Methodology and experimental setup
- Results and performance analysis
- Discussion of Amdahl's Law implications
- Conclusions and future work directions

## 👥 Authors

- **Pranshu Saraswat** - M.S. Ramaiah Institute of Technology
- **Sharanya Sandeep** - M.S. Ramaiah Institute of Technology

## 📜 License

This project is for academic and educational purposes.

## 🔗 References

Key references include:
- AMD Ryzen 9 5900HX technical documentation
- Python multiprocessing documentation
- Embree ray tracing kernels (Wald et al., 2014)
- Amdahl's Law in the Multicore Era (Hill & Marty, 2008)

See `paper.tex` for complete bibliography.
