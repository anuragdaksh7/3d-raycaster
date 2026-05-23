import math
import os
from pathlib import Path

import numpy as np
import pygame

from Cam import Camera
from Light import Light
from Ray import Ray
from sphere import Plane, Sphere


def load_env(path=".env"):
    values = {}
    env_path = Path(path)
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def env_int(values, key, default):
    try:
        return int(values.get(key, default))
    except (TypeError, ValueError):
        return default


def env_float(values, key, default):
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return default


def env_str(values, key, default):
    value = values.get(key, default)
    if value is None:
        return default
    return str(value)


def clamp01(color):
    return np.clip(color, 0.0, 1.0)


def reflect(direction, normal):
    return direction - 2.0 * np.dot(direction, normal) * normal


def background_color(direction):
    t = 0.5 * (direction[1] + 1.0)
    sky = np.array([0.53, 0.74, 0.98], dtype=float)
    horizon = np.array([0.08, 0.09, 0.12], dtype=float)
    return horizon * (1.0 - t) + sky * t


def build_sample_scene():
    objects = [
        Sphere(-1.35, -0.15, 4.25, 0.85, (0.96, 0.34, 0.26), reflectivity=0.12, specular=96),
        Sphere(1.15, -0.35, 5.2, 0.95, (0.28, 0.55, 0.94), reflectivity=0.25, specular=128),
        Sphere(0.15, 0.9, 3.35, 0.45, (0.94, 0.82, 0.22), reflectivity=0.05, specular=48),
        Plane(
            point=(0.0, -1.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            color=(0.82, 0.82, 0.86),
            secondary_color=(0.16, 0.16, 0.18),
            checker_size=1.0,
            reflectivity=0.06,
            specular=12,
        ),
    ]
    lights = [
        Light(5.0, 7.0, -1.5, 1.8, (1.0, 0.98, 0.93)),
        Light(-5.0, 4.5, 1.5, 0.8, (0.6, 0.72, 1.0)),
    ]
    return objects, lights


def find_closest_hit(ray, objects):
    closest_t = None
    closest_object = None
    for obj in objects:
        t = obj.intersect(ray)
        if t is None:
            continue
        if closest_t is None or t < closest_t:
            closest_t = t
            closest_object = obj
    return closest_object, closest_t


def is_shadowed(point, light, objects, ignore_object=None):
    to_light = light.c - point
    light_distance = np.linalg.norm(to_light)
    if light_distance <= 0:
        return False
    ray = Ray(point[0], point[1], point[2], to_light, (1.0, 1.0, 1.0))
    for obj in objects:
        if obj is ignore_object:
            continue
        t = obj.intersect(ray)
        if t is not None and t < light_distance:
            return True
    return False


def shade(ray, objects, lights, ambient=0.12, depth=0, max_depth=1):
    hit_object, hit_distance = find_closest_hit(ray, objects)
    if hit_object is None or hit_distance is None:
        return background_color(ray.d)

    point = ray.p + hit_distance * ray.d
    normal = hit_object.normal_at(point)
    view_dir = -ray.d
    base_color = np.array(hit_object.color_at(point), dtype=float)
    color = base_color * ambient

    for light in lights:
        if is_shadowed(point + normal * 1e-4, light, objects, hit_object):
            continue
        light_dir = light.c - point
        light_distance = np.linalg.norm(light_dir)
        if light_distance == 0:
            continue
        light_dir = light_dir / light_distance
        attenuation = light.intensity / (1.0 + 0.08 * light_distance * light_distance)
        diffuse = max(np.dot(normal, light_dir), 0.0)
        if diffuse > 0:
            color += base_color * light.color * diffuse * attenuation
        halfway = light_dir + view_dir
        halfway_norm = np.linalg.norm(halfway)
        if halfway_norm > 0:
            halfway = halfway / halfway_norm
            specular_strength = getattr(hit_object, "specular", 32.0)
            specular = max(np.dot(normal, halfway), 0.0) ** specular_strength
            color += light.color * specular * attenuation

    reflectivity = getattr(hit_object, "reflectivity", 0.0)
    if reflectivity > 0.0 and depth < max_depth:
        reflected_dir = reflect(ray.d, normal)
        reflected_ray = Ray(*(point + normal * 1e-4), reflected_dir, (1.0, 1.0, 1.0))
        reflection = shade(reflected_ray, objects, lights, ambient=ambient, depth=depth + 1, max_depth=max_depth)
        color = color * (1.0 - reflectivity) + reflection * reflectivity

    return clamp01(color)


def parse_scene_env():
    values = load_env()
    values.update(os.environ)
    quality_raw = values.get("QUALITY")
    quality = env_str(values, "QUALITY", "low").lower()
    quality_presets = {
        "low": (128, 72),
        "medium": (192, 108),
        "high": (256, 144),
        "ultra": (320, 180),
    }
    use_quality_preset = quality_raw is not None and str(quality_raw).strip() != ""
    default_render_width, default_render_height = quality_presets.get(quality, quality_presets["low"])
    config = {
        "window_width": env_int(values, "WINDOW_WIDTH", 960),
        "window_height": env_int(values, "WINDOW_HEIGHT", 540),
        "quality": quality,
        "render_width": default_render_width if use_quality_preset else env_int(values, "RENDER_WIDTH", default_render_width),
        "render_height": default_render_height if use_quality_preset else env_int(values, "RENDER_HEIGHT", default_render_height),
        "fov": env_float(values, "FOV", 70.0),
        "camera_x": env_float(values, "CAMERA_X", 0.0),
        "camera_y": env_float(values, "CAMERA_Y", 0.15),
        "camera_z": env_float(values, "CAMERA_Z", -6.0),
        "camera_yaw": env_float(values, "CAMERA_YAW", 0.0),
        "camera_pitch": env_float(values, "CAMERA_PITCH", 0.0),
        "move_speed": env_float(values, "MOVE_SPEED", 3.5),
        "look_speed": env_float(values, "LOOK_SPEED", 90.0),
        "ambient": env_float(values, "AMBIENT", 0.12),
        "max_frames": env_int(values, "MAX_FRAMES", 0),
    }
    return config


def render_scene(screen, camera, objects, lights, render_size, ambient):
    render_width, render_height = render_size
    pixels = np.zeros((render_height, render_width, 3), dtype=np.float32)
    basis = camera.get_basis()
    for y in range(render_height):
        for x in range(render_width):
            direction = camera.get_ray_direction(x, y, render_width, render_height, basis)
            ray = Ray(camera.x, camera.y, camera.z, direction, (1.0, 1.0, 1.0))
            pixels[y, x] = shade(ray, objects, lights, ambient=ambient)
    surface = pygame.Surface((render_width, render_height)).convert()
    pixel_array = np.ascontiguousarray((pixels * 255).astype(np.uint8).swapaxes(0, 1))
    pygame.surfarray.blit_array(surface, pixel_array)
    scaled = pygame.transform.scale(surface, screen.get_size())
    screen.blit(scaled, (0, 0))


def main():
    config = parse_scene_env()
    pygame.init()
    pygame.display.set_caption("3D Raycaster")
    screen = pygame.display.set_mode((config["window_width"], config["window_height"]))
    clock = pygame.time.Clock()

    objects, lights = build_sample_scene()
    camera = Camera(
        config["camera_x"],
        config["camera_y"],
        config["camera_z"],
        (config["camera_yaw"], config["camera_pitch"], 0.0),
        fov=config["fov"],
    )

    running = True
    frame_count = 0
    startup_grace_frames = 3
    render_size = (config["render_width"], config["render_height"])
    while running:
        dt = clock.tick(30) / 1000.0
        keys = pygame.key.get_pressed()
        forward, right, up = camera.get_basis()
        move_speed = config["move_speed"] * dt
        look_speed = config["look_speed"] * dt

        if keys[pygame.K_w]:
            camera.move(forward * move_speed)
        if keys[pygame.K_s]:
            camera.move(-forward * move_speed)
        if keys[pygame.K_a]:
            camera.move(-right * move_speed)
        if keys[pygame.K_d]:
            camera.move(right * move_speed)
        if keys[pygame.K_q]:
            camera.move(up * move_speed)
        if keys[pygame.K_e]:
            camera.move(-up * move_speed)
        if keys[pygame.K_LEFT]:
            camera.rotate(yaw_delta=-look_speed)
        if keys[pygame.K_RIGHT]:
            camera.rotate(yaw_delta=look_speed)
        if keys[pygame.K_UP]:
            camera.rotate(pitch_delta=look_speed)
        if keys[pygame.K_DOWN]:
            camera.rotate(pitch_delta=-look_speed)

        render_scene(screen, camera, objects, lights, render_size, config["ambient"])
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT and frame_count >= startup_grace_frames:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        frame_count += 1
        if config["max_frames"] and frame_count >= config["max_frames"]:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()