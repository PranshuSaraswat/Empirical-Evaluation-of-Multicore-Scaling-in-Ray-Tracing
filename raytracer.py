"""
raytracer.py

Usage examples:
  python raytracer.py --scene scenes/scene1.json --width 800 --height 600 --threads 1 --tile 16 --outfile results/out_seq_800x600.png
  python raytracer.py --scene scenes/scene1.json --width 1920 --height 1080 --threads 8 --tile 16 --outfile results/out_par_1080_t8.png

Prints:
  RENDER_TIME_MS: <ms>
  CPU_AVG: <percent>
  CPU_MAX: <percent>
"""
import os
import json
import math
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
from PIL import Image

# New imports for CPU sampling
import psutil
import threading

# -------------------------
# Math utilities
# -------------------------
@dataclass
class Vec3:
    x: float
    y: float
    z: float

    # vector ops
    def __add__(self, other): return Vec3(self.x+other.x, self.y+other.y, self.z+other.z)
    def __sub__(self, other): return Vec3(self.x-other.x, self.y-other.y, self.z-other.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s) if isinstance(s,(int,float)) else Vec3(self.x* s.x, self.y*s.y, self.z*s.z)
    __rmul__ = __mul__
    def __truediv__(self, s): return Vec3(self.x/s, self.y/s, self.z/s)
    def dot(self, other): return self.x*other.x + self.y*other.y + self.z*other.z
    def length(self): return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    def normalize(self):
        l = self.length()
        if l == 0: return Vec3(0,0,0)
        return self / l
    def reflect(self, normal):
        # reflection of this vector around normal
        # assume self is incoming direction
        d = self
        return d - normal * (2 * d.dot(normal))

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def vec_to_rgb(vec):
    # vec components in [0,1]
    r = int(clamp(vec.x,0,1) * 255 + 0.5)
    g = int(clamp(vec.y,0,1) * 255 + 0.5)
    b = int(clamp(vec.z,0,1) * 255 + 0.5)
    return (r,g,b)

# -------------------------
# Scene primitives
# -------------------------
@dataclass
class Ray:
    origin: Vec3
    direction: Vec3

@dataclass
class Sphere:
    center: Vec3
    radius: float
    color: Vec3
    reflect: float = 0.0  # [0..1]

    def intersect(self, ray: Ray):
        # returns (hit, t, point, normal)
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius*self.radius
        disc = b*b - 4*a*c
        if disc < 0:
            return (False, None, None, None)
        sqrt_d = math.sqrt(disc)
        t0 = (-b - sqrt_d) / (2*a)
        t1 = (-b + sqrt_d) / (2*a)
        t = None
        if t0 > 1e-5:
            t = t0
        elif t1 > 1e-5:
            t = t1
        else:
            return (False, None, None, None)
        point = ray.origin + ray.direction * t
        normal = (point - self.center).normalize()
        return (True, t, point, normal)

@dataclass
class Light:
    position: Vec3
    intensity: float = 1.0

# -------------------------
# Camera
# -------------------------
class Camera:
    def __init__(self, look_from, look_at, up, fov_deg, aspect):
        self.pos = look_from
        self.fov = math.radians(fov_deg)
        self.aspect = aspect
        self.w = (look_from - look_at).normalize()
        self.u = up.cross(self.w).normalize() if hasattr(up,'cross') else Vec3(1,0,0)  # fallback
        # build orthonormal basis properly:
        # Recompute u and v robustly:
        tmp = (look_at - look_from).normalize()
        # for a simple pinhole, compute basis:
        self.forward = tmp
        # compute right and up:
        upv = up
        right = cross(self.forward, upv).normalize()
        self.right = right
        self.up = cross(right, self.forward).normalize()
        self.half_height = math.tan(self.fov/2)
        self.half_width = self.aspect * self.half_height

def cross(a: Vec3, b: Vec3):
    return Vec3(a.y*b.z - a.z*b.y,
                a.z*b.x - a.x*b.z,
                a.x*b.y - a.y*b.x)

def get_camera(look_from, look_at, up, fov_deg, width, height):
    aspect = width/height
    cam = type('C', (), {})()
    cam.pos = Vec3(*look_from)
    cam.forward = (Vec3(*look_at) - cam.pos).normalize()
    cam.right = cross(cam.forward, Vec3(*up)).normalize()
    cam.up = cross(cam.right, cam.forward).normalize()
    cam.half_height = math.tan(math.radians(fov_deg)/2)
    cam.half_width = aspect * cam.half_height
    return cam

def get_ray_for_pixel(cam, x, y, width, height):
    # pixel coordinates to NDC -1..1
    u = ( (x + 0.5) / width ) * 2 - 1
    v = 1 - ( (y + 0.5) / height ) * 2  # flip Y
    u *= cam.half_width
    v *= cam.half_height
    dir_vec = (cam.forward + cam.right * u + cam.up * v).normalize()
    return Ray(cam.pos, dir_vec)

# -------------------------
# Scene loader
# -------------------------
def load_scene(path):
    with open(path, 'r') as f:
        j = json.load(f)
    spheres = []
    lights = []
    cam_conf = j.get('camera', {})
    for o in j.get('objects', []):
        if o.get('type','sphere') == 'sphere':
            c = o['center']
            col = o.get('color', [255,0,0])
            spheres.append(Sphere(center=Vec3(*c), radius=float(o['radius']), color=Vec3(col[0]/255.0,col[1]/255.0,col[2]/255.0), reflect=float(o.get('reflect',0.0))))
    for L in j.get('lights', []):
        lights.append(Light(position=Vec3(*L['position']), intensity=float(L.get('intensity',1.0))))
    # default camera if missing
    camera = j.get('camera', {})
    look_from = camera.get('look_from', [0,0,0])
    look_at = camera.get('look_at', [0,0,-1])
    up = camera.get('up', [0,1,0])
    fov = camera.get('fov', 60)
    return spheres, lights, (look_from, look_at, up, fov)

# -------------------------
# Tracing
# -------------------------
def trace_ray(ray, spheres, lights, depth=0, max_depth=2):
    # find nearest intersection
    nearest_t = float('inf')
    hit_obj = None
    hit_point = None
    normal = None
    for s in spheres:
        hit, t, p, n = s.intersect(ray)
        if hit and t < nearest_t:
            nearest_t = t
            hit_obj = s
            hit_point = p
            normal = n
    if hit_obj is None:
        # background color
        return Vec3(0.0, 0.0, 0.0)
    # simple Phong-ish shading
    ambient = 0.05
    color = hit_obj.color * ambient
    view_dir = (ray.direction * -1).normalize()
    for light in lights:
        to_light = (light.position - hit_point)
        dist_to_light = to_light.length()
        l_dir = to_light.normalize()
        # shadow check
        shadow_ray = Ray(hit_point + normal * 1e-4, l_dir)
        in_shadow = False
        for s in spheres:
            h2, t2, _, _ = s.intersect(shadow_ray)
            if h2 and t2 < dist_to_light:
                in_shadow = True
                break
        if not in_shadow:
            # diffuse
            diff = max(0.0, normal.dot(l_dir))
            color = color + (hit_obj.color * diff * light.intensity)
            # specular
            reflect_dir = (l_dir * -1).reflect(normal).normalize()
            spec = max(0.0, view_dir.dot(reflect_dir)) ** 32
            color = color + Vec3(1,1,1) * spec * 0.3
    # reflection
    if hit_obj.reflect > 0 and depth < max_depth:
        refl_dir = ray.direction.reflect(normal).normalize()
        refl_ray = Ray(hit_point + normal * 1e-4, refl_dir)
        refl_color = trace_ray(refl_ray, spheres, lights, depth+1, max_depth)
        color = color * (1 - hit_obj.reflect) + refl_color * hit_obj.reflect
    # clamp
    return Vec3(clamp(color.x), clamp(color.y), clamp(color.z))

# -------------------------
# Renderer (sequential)
# -------------------------
def render_sequential(spheres, lights, camera_conf, width, height, max_depth=2):
    look_from, look_at, up, fov = camera_conf
    cam = get_camera(look_from, look_at, up, fov, width, height)
    buf = np.zeros((height, width, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for y in range(height):
        for x in range(width):
            ray = get_ray_for_pixel(cam, x, y, width, height)
            col = trace_ray(ray, spheres, lights, depth=0, max_depth=max_depth)
            buf[y, x] = vec_to_rgb(col)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    return ms, buf

# -------------------------
# Tile worker for parallel
# -------------------------
def render_tile_worker(args):
    # This runs in a worker process. Unpack everything.
    (x0,x1,y0,y1, scene_data, cam_data, width, height, max_depth) = args
    spheres_j, lights_j = scene_data
    # reconstruct spheres & lights
    spheres = []
    for s in spheres_j:
        spheres.append(Sphere(center=Vec3(*s['center']), radius=float(s['radius']), color=Vec3(s['color'][0],s['color'][1],s['color'][2]), reflect=float(s.get('reflect',0.0))))
    lights = []
    for L in lights_j:
        lights.append(Light(position=Vec3(*L['position']), intensity=float(L.get('intensity',1.0))))
    cam = get_camera(*cam_data, width, height)
    block_h = y1 - y0
    block_w = x1 - x0
    block = np.zeros((block_h, block_w, 3), dtype=np.uint8)
    for j, y in enumerate(range(y0, y1)):
        for i, x in enumerate(range(x0, x1)):
            ray = get_ray_for_pixel(cam, x, y, width, height)
            col = trace_ray(ray, spheres, lights, depth=0, max_depth=max_depth)
            block[j, i] = vec_to_rgb(col)
    return (x0, x1, y0, y1, block)

def make_tiles(width, height, tile):
    tiles = []
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            x1 = min(x+tile, width)
            y1 = min(y+tile, height)
            tiles.append((x, x1, y, y1))
    return tiles

# -------------------------
# Renderer (parallel)
# -------------------------
def render_parallel(spheres, lights, camera_conf, width, height, threads=4, tile=16, max_depth=2):
    look_from, look_at, up, fov = camera_conf
    cam_data = (look_from, look_at, up, fov)
    # pack scene in small dicts for pickling
    spheres_j = []
    for s in spheres:
        spheres_j.append({'center': (s.center.x, s.center.y, s.center.z), 'radius': s.radius, 'color': (s.color.x, s.color.y, s.color.z), 'reflect': s.reflect})
    lights_j = []
    for L in lights:
        lights_j.append({'position': (L.position.x, L.position.y, L.position.z), 'intensity': L.intensity})
    scene_data = (spheres_j, lights_j)
    tiles = make_tiles(width, height, tile)
    buf = np.zeros((height, width, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    # launch processes
    with ProcessPoolExecutor(max_workers=threads) as exe:
        futures = []
        for (x0,x1,y0,y1) in tiles:
            args = (x0,x1,y0,y1, scene_data, cam_data, width, height, max_depth)
            futures.append(exe.submit(render_tile_worker, args))
        for f in as_completed(futures):
            x0,x1,y0,y1,block = f.result()
            buf[y0:y1, x0:x1, :] = block
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    return ms, buf

# -------------------------
# CPU monitoring utilities
# -------------------------
def start_cpu_monitor(interval=0.1):
    """
    Start a background thread sampling psutil.cpu_percent(percpu=True).
    Returns (cpu_samples_list, stop_function).
    cpu_samples_list will be a list of lists (each entry is a sample of per-core percentages).
    Call stop() to terminate sampling.
    """
    cpu_samples = []
    stop_event = threading.Event()

    def monitor():
        # First warm-up call for psutil on some systems (establishes baseline)
        psutil.cpu_percent(interval=None, percpu=True)
        while not stop_event.is_set():
            samples = psutil.cpu_percent(interval=None, percpu=True)
            cpu_samples.append(samples)
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    def stop():
        stop_event.set()
        t.join(timeout=1.0)

    return cpu_samples, stop

# -------------------------
# CLI
# -------------------------
def save_png(buf, path):
    img = Image.fromarray(buf, mode='RGB')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def scene_to_pickleable(scene_spheres, scene_lights):
    # return compact picklable representation (not used here)
    pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--width', type=int, default=800)
    p.add_argument('--height', type=int, default=600)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--tile', type=int, default=16)
    p.add_argument('--max_depth', type=int, default=2)
    p.add_argument('--outfile', type=str, default='out.png')
    p.add_argument('--mode', choices=['auto','sequential','parallel'], default='auto', help='auto picks sequential if threads==1')
    return p.parse_args()

def build_scene_from_json(scene_path):
    spheres, lights, cam_conf = load_scene(scene_path)
    # convert sphere colors to Vec3 already done in load_scene
    return spheres, lights, cam_conf

def main():
    args = parse_args()
    spheres, lights, cam_conf = build_scene_from_json(args.scene)

    # start CPU monitor (only in main process)
    cpu_samples, stop_cpu = start_cpu_monitor(interval=0.1)

    if args.mode == 'sequential' or (args.mode=='auto' and args.threads==1):
        ms, buf = render_sequential(spheres, lights, cam_conf, args.width, args.height, max_depth=args.max_depth)
    else:
        ms, buf = render_parallel(spheres, lights, cam_conf, args.width, args.height, threads=args.threads, tile=args.tile, max_depth=args.max_depth)

    # stop CPU sampling and compute statistics
    stop_cpu()
    # small sleep to ensure last sample added (if any)
    time.sleep(0.02)

    # process cpu_samples into per-core averages
    cpu_avg = 0.0
    cpu_max = 0.0
    if len(cpu_samples) == 0:
        # fallback: take a single instantaneous reading
        one = psutil.cpu_percent(interval=None, percpu=True)
        per_core_avg = list(one)
    else:
        # transpose list of samples -> per-core lists
        per_core_lists = list(zip(*cpu_samples))
        per_core_avg = [sum(core)/len(core) for core in per_core_lists]

    if len(per_core_avg) > 0:
        cpu_avg = sum(per_core_avg)/len(per_core_avg)
        cpu_max = max(per_core_avg)

    # save image
    save_png(buf, args.outfile)
    print(f"RENDER_TIME_MS: {ms:.3f}")
    print(f"CPU_AVG: {cpu_avg:.2f}")
    print(f"CPU_MAX: {cpu_max:.2f}")

if __name__ == '__main__':
    main()
