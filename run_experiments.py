"""
Batch runner to execute raytracer.py for multiple threads and resolutions and log results to CSV.

Usage:
  python run_experiments.py --scene scenes/scene1.json --out results/results.csv
"""

import subprocess, csv, os, argparse, time, statistics

DEFAULT_THREADS = [1,2,4,8,16]
DEFAULT_RES = ["640x360","1280x720","1920x1080","3840x2160"]
DEFAULT_TILE = 16
REPEATS = 3

def run_one(scene, w, h, threads, tile, outfile):
    cmd = [
        "python","raytracer.py",
        "--scene",scene,
        "--width",str(w),
        "--height",str(h),
        "--threads",str(threads),
        "--tile",str(tile),
        "--outfile",outfile,
        "--mode","parallel" if threads>1 else "sequential"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr

    ms = None
    cpu_avg = None
    cpu_max = None

    for line in out.splitlines():
        line = line.strip()

        if line.startswith("RENDER_TIME_MS"):
            try:
                ms = float(line.split(":")[1].strip())
            except:
                pass

        elif line.startswith("CPU_AVG"):
            try:
                cpu_avg = float(line.split(":")[1].strip())
            except:
                pass

        elif line.startswith("CPU_MAX"):
            try:
                cpu_max = float(line.split(":")[1].strip())
            except:
                pass

    return ms, cpu_avg, cpu_max, out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', default='results/results.csv')
    parser.add_argument('--threads', default=",".join(str(t) for t in DEFAULT_THREADS))
    parser.add_argument('--res', default=",".join(DEFAULT_RES))
    parser.add_argument('--tile', type=int, default=DEFAULT_TILE)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # CSV header (updated)
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "threads","width","height","tile","run_index",
            "time_ms","cpu_avg","cpu_max","outfile","stdout"
        ])

    thread_list = [int(x) for x in args.threads.split(",")]
    res_list = [x for x in args.res.split(",")]

    for (w_h) in res_list:
        w, h = map(int, w_h.split("x"))

        for t in thread_list:
            times = []
            for r in range(args.repeats):

                out_file = f"results/out_{w}x{h}_t{t}_tile{args.tile}_r{r}.png"
                print(f"Running: {w}x{h}, threads={t}, repeat={r}")

                ms, cpu_avg, cpu_max, stdout = run_one(
                    args.scene, w, h, t, args.tile, out_file
                )

                if ms is None:
                    print("Warning: failed to parse RENDER_TIME_MS. Using 0.")
                    ms = 0.0

                # If CPU values missing, fallback to 0
                if cpu_avg is None: cpu_avg = 0.0
                if cpu_max is None: cpu_max = 0.0

                # Append row
                with open(args.out, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        t, w, h, args.tile, r,
                        f"{ms:.3f}",
                        f"{cpu_avg:.2f}",
                        f"{cpu_max:.2f}",
                        out_file,
                        stdout.replace("\n","\\n")
                    ])

                times.append(ms)
                time.sleep(0.5)

            median_time = statistics.median(times)
            print(f"Median time for {w}x{h}, t={t}: {median_time:.3f} ms")

    print(f"All done. Results in {args.out}")

if __name__ == '__main__':
    main()
