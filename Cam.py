import math
import numpy as np

from Ray import Ray


def _normalize(vector):
    length = np.linalg.norm(vector)
    if length == 0:
        return vector
    return vector / length


class Camera:
    def __init__(self, x, y, z, angle=(0.0, 0.0, 0.0), fov=70.0):
        self.position = np.array([x, y, z], dtype=float)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        if isinstance(angle, (tuple, list)):
            self.yaw = float(angle[0]) if len(angle) > 0 else 0.0
            self.pitch = float(angle[1]) if len(angle) > 1 else 0.0
        else:
            self.yaw = float(angle)
            self.pitch = 0.0
        self.roll = float(angle[2]) if isinstance(angle, (tuple, list)) and len(angle) > 2 else 0.0
        self.fov = float(fov)
        self.rays = []

    def get_basis(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)

        forward = np.array(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ],
            dtype=float,
        )
        forward = _normalize(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = _normalize(np.cross(world_up, forward))
        if np.linalg.norm(right) == 0:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        up = _normalize(np.cross(forward, right))
        return forward, right, up

    def get_ray(self, px, py, width, height):
        forward, right, up = self.get_basis()
        aspect_ratio = width / height
        scale = math.tan(math.radians(self.fov) / 2.0)
        x_ndc = (2.0 * (px + 0.5) / width) - 1.0
        y_ndc = 1.0 - (2.0 * (py + 0.5) / height)
        direction = forward + x_ndc * aspect_ratio * scale * right + y_ndc * scale * up
        return Ray(self.x, self.y, self.z, direction, (1.0, 1.0, 1.0))

    def emitRays(self, w, h, q_factor=1):
        self.rays = [[self.get_ray(j, i, w, h) for j in range(w)] for i in range(h)]
        return self.rays

    def move(self, delta):
        delta = np.array(delta, dtype=float)
        self.position += delta
        self.x, self.y, self.z = self.position.tolist()

    def rotate(self, yaw_delta=0.0, pitch_delta=0.0):
        self.yaw = (self.yaw + yaw_delta) % 360.0
        self.pitch = max(-89.0, min(89.0, self.pitch + pitch_delta))


if __name__ == "__main__":
    camera = Camera(0, 0, 0, (0, 0, 0))
    print(camera.get_ray(0, 0, 10, 10).d)