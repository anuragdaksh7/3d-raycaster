import math
import os
from pathlib import Path

import numpy as np
import pygame

try:
    import cupy as cp
except ImportError:
    cp = None

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


def env_bool(values, key, default=False):
    value = values.get(key)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def choose_backend(values):
    requested = env_str(values, "RENDER_BACKEND", "auto").strip().lower()
    if requested in {"cuda", "gpu"}:
        return (cp, "cuda") if cp is not None else (np, "cpu")
    if requested == "auto" and cp is not None:
        return cp, "cuda"
    return np, "cpu"


def xp_array(xp, value):
    return xp.asarray(value, dtype=xp.float32)


def normalize_vectors(vectors, xp):
    lengths = xp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / xp.maximum(lengths, 1e-8)


class SceneRenderer:
    def __init__(self, objects, lights, width, height, fov, ambient, xp, enable_shadows=True, aa_samples=1):
        self.objects = objects
        self.lights = lights
        self.width = width
        self.height = height
        self.fov = fov
        self.ambient = ambient
        self.xp = xp
        self.enable_shadows = enable_shadows
        self.aa_samples = max(1, int(aa_samples))
        self.surface = pygame.Surface((width, height)).convert()
        self.scale = math.tan(math.radians(fov) / 2.0)
        aspect_ratio = width / height
        x_coords = (((np.arange(width) + 0.5) / width) * 2.0 - 1.0) * aspect_ratio * self.scale
        y_coords = (1.0 - ((np.arange(height) + 0.5) / height) * 2.0) * self.scale
        self.x_coords = xp_array(xp, x_coords)
        self.y_coords = xp_array(xp, y_coords)
        self.sky_top = xp_array(xp, (0.53, 0.74, 0.98))
        self.sky_bottom = xp_array(xp, (0.08, 0.09, 0.12))
        self.light_positions = [xp_array(xp, light.c) for light in lights]
        self.light_colors = [xp_array(xp, light.color) for light in lights]
        self.light_intensities = [float(light.intensity) for light in lights]
        side = int(math.ceil(math.sqrt(self.aa_samples)))
        offsets = []
        for row in range(side):
            for col in range(side):
                if len(offsets) >= self.aa_samples:
                    break
                dx = (col + 0.5) / side - 0.5
                dy = (row + 0.5) / side - 0.5
                offsets.append((dx, dy))
        self.sample_offsets = xp_array(xp, offsets)

    def build_directions(self, camera):
        forward, right, up = camera.get_basis()
        forward = xp_array(self.xp, forward)
        right = xp_array(self.xp, right)
        up = xp_array(self.xp, up)
        directions = (
            forward[None, None, :]
            + self.x_coords[None, :, None] * right[None, None, :]
            + self.y_coords[:, None, None] * up[None, None, :]
        )
        return normalize_vectors(directions, self.xp)

    def build_sample_directions(self, camera, offset_x, offset_y):
        forward, right, up = camera.get_basis()
        forward = xp_array(self.xp, forward)
        right = xp_array(self.xp, right)
        up = xp_array(self.xp, up)
        x_coords = self.x_coords + offset_x * (2.0 * self.scale / self.width)
        y_coords = self.y_coords + offset_y * (2.0 * self.scale / self.height)
        directions = (
            forward[None, None, :]
            + x_coords[None, :, None] * right[None, None, :]
            + y_coords[:, None, None] * up[None, None, :]
        )
        return normalize_vectors(directions, self.xp)

    def background_for(self, directions):
        t = 0.5 * (directions[..., 1] + 1.0)
        return self.sky_bottom * (1.0 - t)[..., None] + self.sky_top * t[..., None]

    def intersect_sphere(self, origin, directions, sphere):
        center = xp_array(self.xp, (sphere.x, sphere.y, sphere.z))
        oc = origin - center
        b = self.xp.sum(oc * directions, axis=-1)
        c = self.xp.sum(oc * oc, axis=-1) - sphere.radius * sphere.radius
        discriminant = b * b - c
        sqrt_discriminant = self.xp.sqrt(self.xp.maximum(discriminant, 0.0))
        t1 = -b - sqrt_discriminant
        t2 = -b + sqrt_discriminant
        valid = discriminant >= 0.0
        return self.xp.where(
            valid & (t1 > 1e-4),
            t1,
            self.xp.where(valid & (t2 > 1e-4), t2, self.xp.inf),
        )

    def intersect_plane(self, origin, directions, plane):
        point = xp_array(self.xp, plane.point)
        normal = xp_array(self.xp, plane.normal)
        denom = self.xp.sum(directions * normal, axis=-1)
        numerator = self.xp.sum((point - origin) * normal, axis=-1)
        t = numerator / denom
        valid = (self.xp.abs(denom) > 1e-4) & (t > 1e-4)
        return self.xp.where(valid, t, self.xp.inf)

    def intersect_object(self, origin, directions, obj):
        if isinstance(obj, Sphere):
            return self.intersect_sphere(origin, directions, obj)
        if isinstance(obj, Plane):
            return self.intersect_plane(origin, directions, obj)
        raise TypeError(f"Unsupported object type: {type(obj)!r}")

    def trace_scene(self, origin, directions):
        hits = [self.intersect_object(origin, directions, obj) for obj in self.objects]
        if not hits:
            shape = directions.shape[:2]
            return self.xp.full(shape, self.xp.inf), self.xp.full(shape, -1, dtype=self.xp.int32)
        hit_stack = self.xp.stack(hits, axis=0)
        hit_index = self.xp.argmin(hit_stack, axis=0).astype(self.xp.int32)
        closest_t = self.xp.min(hit_stack, axis=0)
        return closest_t, hit_index

    def plane_color(self, points, plane):
        pattern = self.xp.floor(points[..., 0] / plane.checker_size) + self.xp.floor(points[..., 2] / plane.checker_size)
        checker = (pattern.astype(self.xp.int32) & 1) == 0
        primary = xp_array(self.xp, plane.color)
        secondary = xp_array(self.xp, plane.secondary_color)
        return self.xp.where(checker[..., None], primary, secondary)

    def shadow_visibility(self, points, normals, light_position, object_index):
        to_light = light_position - (points + normals * 1e-4)
        distances = self.xp.linalg.norm(to_light, axis=-1)
        light_dirs = to_light / self.xp.maximum(distances[..., None], 1e-8)
        occluded = self.xp.zeros(distances.shape, dtype=bool)
        for index, obj in enumerate(self.objects):
            if index == object_index:
                continue
            shadow_hit = self.intersect_object(points + normals * 1e-4, light_dirs, obj)
            occluded |= (shadow_hit > 1e-4) & (shadow_hit < distances)
        return ~occluded, light_dirs, distances

    def shade_object(self, points, directions, obj, object_index):
        if isinstance(obj, Sphere):
            center = xp_array(self.xp, (obj.x, obj.y, obj.z))
            normals = normalize_vectors(points - center, self.xp)
            base_color = self.xp.broadcast_to(xp_array(self.xp, obj.color), points.shape)
        else:
            normals = self.xp.broadcast_to(xp_array(self.xp, obj.normal), points.shape)
            base_color = self.plane_color(points, obj)

        color = base_color * self.ambient
        view_dirs = -directions
        metallic = float(getattr(obj, "metallic", 0.0))
        roughness = float(getattr(obj, "roughness", 0.0))
        emissive = float(getattr(obj, "emissive", 0.0))

        for light_position, light_color, light_intensity in zip(self.light_positions, self.light_colors, self.light_intensities):
            if self.enable_shadows:
                visible, light_dirs, distances = self.shadow_visibility(points, normals, light_position, object_index)
            else:
                to_light = light_position - points
                distances = self.xp.linalg.norm(to_light, axis=-1)
                light_dirs = to_light / self.xp.maximum(distances[..., None], 1e-8)
                visible = self.xp.ones(distances.shape, dtype=bool)

            attenuation = light_intensity / (1.0 + 0.08 * distances * distances)
            diffuse = self.xp.maximum(self.xp.sum(normals * light_dirs, axis=-1), 0.0)
            diffuse_color = base_color * (1.0 - metallic)
            color += diffuse_color * light_color * diffuse[..., None] * attenuation[..., None] * visible[..., None]

            halfway = normalize_vectors(light_dirs + view_dirs, self.xp)
            specular_strength = max(4.0, getattr(obj, "specular", 32.0) * (1.0 - 0.65 * roughness))
            specular = self.xp.maximum(self.xp.sum(normals * halfway, axis=-1), 0.0) ** specular_strength
            specular_color = light_color * (1.0 - metallic) + base_color * metallic
            color += specular_color * specular[..., None] * attenuation[..., None] * visible[..., None]

        if emissive > 0.0:
            color += base_color * emissive

        return self.xp.clip(color, 0.0, 1.0)

    def render(self, camera):
        origin = xp_array(self.xp, (camera.x, camera.y, camera.z))
        if self.aa_samples <= 1:
            directions = self.build_directions(camera)
            image = self._render_directions(origin, directions)
            self.present(image)
            return

        image = self.xp.zeros((self.height, self.width, 3), dtype=self.xp.float32)
        for offset_x, offset_y in self.sample_offsets:
            directions = self.build_sample_directions(camera, float(offset_x), float(offset_y))
            image += self._render_directions(origin, directions)
        image /= float(self.aa_samples)
        self.present(self.xp.clip(image, 0.0, 1.0))

    def _render_directions(self, origin, directions):
        image = self.background_for(directions)
        closest_t, hit_index = self.trace_scene(origin, directions)
        hit_mask = self.xp.isfinite(closest_t)
        if not self.xp.any(hit_mask):
            return image

        points = origin[None, None, :] + closest_t[..., None] * directions
        for object_index, obj in enumerate(self.objects):
            object_mask = (hit_index == object_index) & hit_mask
            if not self.xp.any(object_mask):
                continue
            shaded = self.shade_object(points[object_mask], directions[object_mask], obj, object_index)
            image[object_mask] = shaded

        return self.xp.clip(image, 0.0, 1.0)

    def present(self, image):
        if self.xp is np:
            pixels = (image * 255).astype(np.uint8).swapaxes(0, 1)
        else:
            pixels = cp.asnumpy((image * 255).astype(cp.uint8)).swapaxes(0, 1)
        pygame.surfarray.blit_array(self.surface, np.ascontiguousarray(pixels))


def build_sample_scene():
    objects = [
        Sphere(-1.45, -0.15, 4.15, 0.78, (0.96, 0.34, 0.26), roughness=0.85, reflectivity=0.04, specular=24, metallic=0.0),
        Sphere(1.05, -0.35, 5.0, 0.92, (0.28, 0.55, 0.94), roughness=0.18, reflectivity=0.22, specular=128, metallic=0.35),
        Sphere(0.2, 0.82, 3.25, 0.42, (0.94, 0.82, 0.22), roughness=0.08, reflectivity=0.08, specular=192, metallic=0.78),
        Plane(
            point=(0.0, -1.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            color=(0.82, 0.82, 0.86),
            secondary_color=(0.16, 0.16, 0.18),
            checker_size=1.0,
            roughness=1.0,
            reflectivity=0.04,
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
    backend, backend_name = choose_backend(values)
    config = {
        "window_width": env_int(values, "WINDOW_WIDTH", 960),
        "window_height": env_int(values, "WINDOW_HEIGHT", 540),
        "quality": quality,
        "backend": backend,
        "backend_name": backend_name,
        "render_width": default_render_width if use_quality_preset else env_int(values, "RENDER_WIDTH", default_render_width),
        "render_height": default_render_height if use_quality_preset else env_int(values, "RENDER_HEIGHT", default_render_height),
        "aa_samples": env_int(values, "AA_SAMPLES", 1),
        "fov": env_float(values, "FOV", 70.0),
        "camera_x": env_float(values, "CAMERA_X", 0.0),
        "camera_y": env_float(values, "CAMERA_Y", 0.15),
        "camera_z": env_float(values, "CAMERA_Z", -6.0),
        "camera_yaw": env_float(values, "CAMERA_YAW", 0.0),
        "camera_pitch": env_float(values, "CAMERA_PITCH", 0.0),
        "move_speed": env_float(values, "MOVE_SPEED", 3.5),
        "look_speed": env_float(values, "LOOK_SPEED", 90.0),
        "ambient": env_float(values, "AMBIENT", 0.12),
        "enable_shadows": env_bool(values, "ENABLE_SHADOWS", False),
        "max_frames": env_int(values, "MAX_FRAMES", 0),
    }
    return config


def main():
    config = parse_scene_env()
    pygame.init()
    pygame.display.set_caption(f"3D Raycaster [{config['backend_name']}]")
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
    renderer = SceneRenderer(
        objects,
        lights,
        config["render_width"],
        config["render_height"],
        config["fov"],
        config["ambient"],
        config["backend"],
        enable_shadows=config["enable_shadows"],
        aa_samples=config["aa_samples"],
    )

    running = True
    frame_count = 0
    startup_grace_frames = 3
    while running:
        dt = clock.tick(60) / 1000.0
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

        renderer.render(camera)
        screen.blit(pygame.transform.scale(renderer.surface, screen.get_size()), (0, 0))
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