"""Meshes, textures and static scenery. Every shape here is visual only.

Meshes are generated at startup so that the repository needs no binary assets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

FLOOR_Z = -0.40
TABLE_TOP_Z = 0.32
TABLE_THICKNESS_M = 0.03
TRAY_HEIGHT_M = 0.008
DISH_RADIUS_M = 0.062
DISH_HEIGHT_M = 0.028
TEXTURE_SIZE = 512
LATHE_SEGMENTS = 48
MONITOR_PIXELS = (320, 240)
STATUS_PIXELS = (640, 310)
IDENTITY = (0.0, 0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0)
CERAMIC = (0.93, 0.92, 0.88)
LACQUER = (0.30, 0.07, 0.05)
STEEL = (0.78, 0.79, 0.82)
WALL = (0.90, 0.87, 0.80)
SHINY = (0.8, 0.8, 0.8)
MATTE = (0.05, 0.05, 0.05)


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray

    def write(self, path: Path) -> None:
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.vertices]
        lines += [f"vt {u:.5f} {v:.5f}" for u, v in self.uvs]
        lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in self.normals]
        lines += ["f " + " ".join(f"{i}/{i}/{i}" for i in face + 1) for face in self.faces]
        path.write_text("\n".join(lines) + "\n")


def _smooth_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    face_normals = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def lathe(profile: Sequence[tuple[float, float]], *, lumps: float = 0.0) -> Mesh:
    """Revolve (radius, z) points, listed bottom to top on the outside, around the z axis."""
    rings = len(profile)
    angles = np.linspace(0, 2 * math.pi, LATHE_SEGMENTS + 1)
    radii = np.array([r for r, _ in profile])[:, None] * np.ones(angles.size)
    heights = np.array([z for _, z in profile])[:, None] * np.ones(angles.size)
    if lumps:
        rng = np.random.default_rng(len(profile))
        noise = rng.normal(0, lumps, radii.shape)
        noise[:, -1] = noise[:, 0]
        radii = radii * (1 + noise)
    vertices = np.stack((radii * np.cos(angles), radii * np.sin(angles), heights), axis=-1).reshape(
        -1, 3
    )
    width = angles.size
    uvs = np.stack(
        np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, rings)), axis=-1
    ).reshape(-1, 2)
    faces = []
    for ring in range(rings - 1):
        for column in range(width - 1):
            low, high = ring * width + column, (ring + 1) * width + column
            faces += [(low, low + 1, high + 1), (low, high + 1, high)]
    faces = np.array(faces)
    normals = _smooth_normals(vertices, faces)
    seam_low, seam_high = np.arange(rings) * width, np.arange(rings) * width + width - 1
    seam = normals[seam_low] + normals[seam_high]
    seam /= np.maximum(np.linalg.norm(seam, axis=1, keepdims=True), 1e-12)
    normals[seam_low] = normals[seam_high] = seam
    return Mesh(vertices, normals, uvs, faces)


def dome(z_from: float = -1.0, z_to: float = 1.0, *, lumps: float = 0.0) -> Mesh:
    """A unit sphere, or a slice of one between two heights."""
    start, stop = math.asin(z_from), math.asin(z_to)
    angles = np.linspace(start, stop, 25)
    return lathe([(max(math.cos(a), 1e-4), math.sin(a)) for a in angles], lumps=lumps)


def rounded_cylinder(radius: float, height: float, edge: float) -> Mesh:
    arc = [i * math.pi / 12 for i in range(7)]
    bottom = [(radius - edge * (1 - math.sin(a)), edge * (1 - math.cos(a))) for a in arc]
    top = [(r, height - z) for r, z in reversed(bottom)]
    return lathe([(1e-4, 0.0), *bottom, *top, (1e-4, height)])


def dish() -> Mesh:
    """A shallow ceramic dish with an inside and an outside."""
    outer = [(1e-4, 0.0), (0.030, 0.0), (0.033, 0.002)]
    outer += [(0.050, 0.012), (DISH_RADIUS_M - 0.002, 0.024), (DISH_RADIUS_M, DISH_HEIGHT_M)]
    inner = [(DISH_RADIUS_M - 0.003, DISH_HEIGHT_M), (0.048, 0.016), (0.030, 0.007), (1e-4, 0.006)]
    return lathe(outer + inner)


def box(size: tuple[float, float, float], texture_repeat: float = 1.0) -> Mesh:
    """An axis-aligned box centered at the origin, with flat faces."""
    half = np.array(size) / 2
    vertices, normals, uvs, faces = [], [], [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
            normal = np.zeros(3)
            normal[axis] = sign
            base = len(vertices)
            for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                point = np.zeros(3)
                point[axis], point[u_axis], point[v_axis] = sign, u, v
                vertices.append(point * half)
                normals.append(normal)
                uvs.append(((u + 1) / 2 * texture_repeat, (v + 1) / 2 * texture_repeat))
            quad = (0, 1, 2, 0, 2, 3) if sign > 0 else (0, 2, 1, 0, 3, 2)
            faces += [tuple(base + i for i in quad[j : j + 3]) for j in (0, 3)]
    return Mesh(*map(np.array, (vertices, normals, uvs, faces)))


def screen(width: float, height: float) -> Mesh:
    """A flat rectangle facing +x, with the image upright when viewed from the front."""
    half_w, half_h = width / 2, height / 2
    vertices = np.array(
        (
            (0.0, -half_w, -half_h),
            (0.0, half_w, -half_h),
            (0.0, half_w, half_h),
            (0.0, -half_w, half_h),
        )
    )
    normals = np.tile((1.0, 0.0, 0.0), (4, 1))
    uvs = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    return Mesh(vertices, normals, uvs, np.array(((0, 1, 2), (0, 2, 3))))


def spoon(length: float, bowl_half_length: float = 0.03) -> Mesh:
    """A spoon along +x from the grip to the bowl center at length, hollow side on local -z."""
    stations = np.linspace(-0.05, length + bowl_half_length, 70)
    across = np.linspace(-1, 1, 13)
    rows = []
    for x in stations:
        offset = (x - length) / bowl_half_length
        in_bowl = abs(offset) < 1
        bowl_width = 0.02 * math.sqrt(max(0.0, 1 - offset**2))
        width = max(0.0035 + 0.002 * max(0.0, -x) / 0.05, bowl_width) if x < length else bowl_width
        depth = 0.008 * (1 - offset**2) if in_bowl else 0.0
        rows.append([(x, width * s, depth * (1 - s**2)) for s in across])
    grid = np.array(rows)
    columns = across.size
    vertices = np.concatenate((grid.reshape(-1, 3), (grid + (0, 0, -0.0012)).reshape(-1, 3)))
    uvs = np.zeros((vertices.shape[0], 2))
    faces = []
    count = columns * stations.size
    for row in range(stations.size - 1):
        for column in range(columns - 1):
            a, b = row * columns + column, row * columns + column + 1
            c, d = a + columns, b + columns
            faces += [(a, c, d), (a, d, b)]
            faces += [(a + count, d + count, c + count), (a + count, b + count, d + count)]
    faces = np.array(faces)
    return Mesh(vertices, _smooth_normals(vertices, faces), uvs, faces)


def wood_texture(path: Path, base: tuple[float, float, float], seed: int = 1) -> None:
    """Write a procedural plank texture with grain along the image's x axis."""
    rng = np.random.default_rng(seed)
    size = TEXTURE_SIZE
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    planks = 6
    plank = np.minimum(np.floor(y * planks), planks - 1)
    shifts = rng.uniform(0, 40, planks)[plank.astype(int)]
    wobble = np.sin(x * 7 + shifts) * 0.25 + np.sin(x * 19 + shifts * 2) * 0.08
    grain = np.sin((y * planks - plank + wobble * 0.15) * 90 + shifts)
    shade = 1 + 0.07 * grain + rng.uniform(-0.06, 0.06, planks)[plank.astype(int)]
    edge = np.abs(y * planks - plank - 0.5) > 0.492
    shade = np.where(edge, shade * 0.72, shade)
    image = np.clip(shade[..., None] * np.array(base), 0, 1)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


class Scenery:
    """Places visual-only bodies, writing generated meshes to a working directory."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.bodies: list[int] = []

    def place(
        self,
        name: str,
        mesh: Mesh,
        color: Sequence[float],
        position: Sequence[float],
        *,
        orientation: Sequence[float] = IDENTITY,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        specular: Sequence[float] = MATTE,
        texture: int | None = None,
    ) -> int:
        path = self._directory / f"{name}.obj"
        if not path.exists():
            mesh.write(path)
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(path),
            meshScale=scale,
            rgbaColor=(*color, 1),
            specularColor=specular,
        )
        body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=orientation,
        )
        if texture is not None:
            p.changeVisualShape(body, -1, textureUniqueId=texture)
        self.bodies.append(body)
        return body

    def ball(self, name, color, center, radii, **options) -> int:
        return self.place(name, dome(), color, center, scale=radii, **options)

    def _texture(self, name: str, base: tuple[float, float, float], seed: int) -> int:
        path = self._directory / f"{name}.png"
        wood_texture(path, base, seed)
        return p.loadTexture(str(path))

    def build_room(self) -> None:
        floor = self._texture("floor", (0.55, 0.39, 0.25), 2)
        z = FLOOR_Z - 0.01
        self.place("floor", box((6, 6, 0.02), 3.0), WHITE, (0, 0, z), texture=floor)
        wall_height = 2.6
        wall_z = FLOOR_Z + wall_height / 2
        self.place("wall_x", box((0.05, 6, wall_height)), WALL, (-1.5, 0, wall_z))
        self.place("wall_y", box((6, 0.05, wall_height)), WALL, (0, 2.0, wall_z))
        self.place("wall_ny", box((6, 0.05, wall_height)), WALL, (0, -2.0, wall_z))
        trim_z = FLOOR_Z + 0.06
        trim = (0.97, 0.96, 0.93)
        self.place("trim_x", box((0.06, 6, 0.12)), trim, (-1.47, 0, trim_z))
        self.place("trim_y", box((6, 0.06, 0.12)), trim, (0, 1.97, trim_z))

    def build_table(self, center: Sequence[float], half_extents: Sequence[float]) -> None:
        wood = self._texture("table", (0.68, 0.49, 0.31), 3)
        cx, cy = center[:2]
        hx, hy = half_extents[:2]
        top = (2 * hx, 2 * hy, TABLE_THICKNESS_M)
        z = TABLE_TOP_Z - TABLE_THICKNESS_M / 2
        self.place("tabletop", box(top), WHITE, (cx, cy, z), texture=wood)
        leg_height = TABLE_TOP_Z - TABLE_THICKNESS_M - FLOOR_Z
        leg = box((0.06, 0.06, leg_height))
        legs = [(sx, sy) for sx in (-1, 1) for sy in (-1, 1)]
        for index, (sx, sy) in enumerate(legs):
            position = (cx + sx * (hx - 0.06), cy + sy * (hy - 0.06), FLOOR_Z + leg_height / 2)
            self.place(f"leg{index}", leg, (0.42, 0.29, 0.18), position)

    def build_robot_stand(self) -> None:
        height = -FLOOR_Z
        stand = rounded_cylinder(0.13, height - 0.03, 0.01)
        self.place("stand", stand, (0.28, 0.29, 0.31), (0, 0, FLOOR_Z), specular=SHINY)
        self.place(
            "stand_top", rounded_cylinder(0.11, 0.03, 0.005), (0.2, 0.2, 0.22), (0, 0, -0.03)
        )

    def build_tray(self, bounds: tuple[float, float, float, float]) -> None:
        x_low, x_high, y_low, y_high = bounds
        size = (x_high - x_low, y_high - y_low, TRAY_HEIGHT_M)
        center = ((x_low + x_high) / 2, (y_low + y_high) / 2, TABLE_TOP_Z + TRAY_HEIGHT_M / 2)
        self.place("tray", box(size), LACQUER, center, specular=SHINY)

    def build_dish(self, food_id: str, style: str, color, pos: Sequence[float]) -> None:
        x, y = pos[:2]
        floor = TABLE_TOP_Z + TRAY_HEIGHT_M
        inside = floor + 0.006
        self.place(f"dish_{food_id}", dish(), CERAMIC, (x, y, floor), specular=SHINY)
        builder = getattr(self, f"_food_{style}")
        builder(food_id, color, x, y, inside)

    def _food_mound(self, food_id, color, x, y, z) -> None:
        mound = dome(0.0, 1.0, lumps=0.05)
        self.place(f"food_{food_id}", mound, color, (x, y, z), scale=(0.05, 0.05, 0.035))

    def _food_soup(self, food_id, color, x, y, z) -> None:
        liquid = rounded_cylinder(0.052, 0.012, 0.002)
        self.place(f"food_{food_id}", liquid, color, (x, y, z), specular=SHINY)
        for index, (dx, dy) in enumerate(((0.02, 0.01), (-0.015, 0.02), (0.0, -0.022))):
            cube = box((0.012, 0.012, 0.006))
            orientation = p.getQuaternionFromEuler((0, 0, index))
            position = (x + dx, y + dy, z + 0.012)
            self.place(
                f"tofu{food_id}{index}", cube, (0.95, 0.93, 0.85), position, orientation=orientation
            )

    def _food_chunks(self, food_id, color, x, y, z) -> None:
        chunk = rounded_cylinder(0.016, 0.014, 0.004)
        spots = ((0.0, 0.0, 0.017), (0.024, 0.0, 0.0), (-0.022, 0.01, 0.0))
        spots += ((0.0, 0.024, 0.0), (0.006, -0.024, 0.0))
        for index, (dx, dy, dz) in enumerate(spots):
            orientation = p.getQuaternionFromEuler((0.35 * (index - 2), 0.3 * index, index))
            position = (x + dx, y + dy, z + dz)
            self.place(f"chunk{index}", chunk, color, position, orientation=orientation)

    def _food_pudding(self, food_id, color, x, y, z) -> None:
        body = lathe([(1e-4, 0), (0.042, 0), (0.042, 0.002), (0.030, 0.030), (1e-4, 0.030)])
        self.place(f"food_{food_id}", body, color, (x, y, z), specular=SHINY)
        caramel = rounded_cylinder(0.031, 0.004, 0.0015)
        self.place("caramel", caramel, (0.36, 0.17, 0.05), (x, y, z + 0.029), specular=SHINY)

    def build_monitor(
        self, name: str, center: np.ndarray, normal: np.ndarray, size, pixels: tuple[int, int]
    ) -> int:
        """Place a screen facing along normal, and return the texture that shows on it."""
        width, height = size
        yaw = math.atan2(normal[1], normal[0])
        pitch = math.atan2(-normal[2], math.hypot(normal[0], normal[1]))
        orientation = p.getQuaternionFromEuler((0.0, pitch, yaw))
        depth = 0.004
        border = height * 0.025
        frame_size = (depth, width + border, height + border)
        frame_center = center - normal * depth
        self.place(
            "monitor_frame",
            box(frame_size),
            (0.06, 0.06, 0.07),
            frame_center,
            orientation=orientation,
        )
        path = self._directory / f"{name}.png"
        black = np.zeros((pixels[1], pixels[0], 3), dtype=np.uint8)
        Image.fromarray(black).save(path)
        texture = p.loadTexture(str(path))
        self.place(
            f"{name}_screen",
            screen(width, height),
            WHITE,
            center,
            orientation=orientation,
            texture=texture,
        )
        return texture

    def build_person(self, person: dict) -> None:
        mouth = np.array(person["mouth"], dtype=float)
        skin, shirt = person["skin_color"], person["shirt_color"]
        hair, trousers = person["hair_color"], person["trousers_color"]
        head = mouth + (0, 0.088, 0.035)
        first_head_body = len(self.bodies)
        self.ball("head", skin, head, (0.075, 0.09, 0.105))
        tilt = p.getQuaternionFromEuler((-0.28, 0, 0))
        cap = dome(-0.05, 1.0)
        self.place(
            "hair",
            cap,
            hair,
            head + (0, 0.006, 0.006),
            orientation=tilt,
            scale=(0.079, 0.094, 0.109),
        )
        self.ball("lip", (0.72, 0.36, 0.34), mouth + (0, 0.003, 0), (0.021, 0.006, 0.008))
        self.ball("mouth_open", (0.22, 0.06, 0.06), mouth + (0, -0.0035, 0), (0.014, 0.002, 0.004))
        self.ball("nose", skin, mouth + (0, -0.006, 0.042), (0.012, 0.014, 0.017))
        for side, name in ((-1, "l"), (1, "r")):
            eye = mouth + (side * 0.031, 0.021, 0.074)
            self.ball(f"eye{name}", (0.97, 0.97, 0.96), eye, (0.012, 0.006, 0.007))
            self.ball(
                f"iris{name}", (0.10, 0.07, 0.06), eye + (0, -0.005, 0), (0.0065, 0.003, 0.0065)
            )
            brow = mouth + (side * 0.032, 0.015, 0.094)
            self.ball(f"brow{name}", hair, brow, (0.016, 0.004, 0.003))
            self.ball(
                f"ear{name}", skin, head + (side * 0.076, 0.005, -0.002), (0.009, 0.019, 0.028)
            )

        neck_top = head - (0, -0.015, 0.075)
        self.place("neck", rounded_cylinder(0.042, 0.10, 0.005), skin, neck_top - (0, 0, 0.09))
        self.head_bodies = self.bodies[first_head_body:]
        waist_z = FLOOR_Z + 0.50
        shoulder_z = neck_top[2] - 0.055
        torso_y = neck_top[1] + 0.015
        height = shoulder_z - waist_z
        profile = [(1e-4, 0.0), (0.92, 0.0), (0.98, 0.15), (0.92, 0.45), (1.0, 0.72)]
        profile += [(1.0, 0.86), (0.72, 0.97), (0.26, 1.03), (1e-4, 1.04)]
        torso = lathe(profile)
        scale = (0.175, 0.105, height)
        self.place("torso", torso, shirt, (mouth[0], torso_y, waist_z), scale=scale)

        for side, name in ((-1, "l"), (1, "r")):
            x = mouth[0] + side * 0.185
            shoulder = np.array((x, torso_y, shoulder_z - 0.02))
            self.ball(f"shoulder{name}", shirt, shoulder, (0.046, 0.05, 0.05))
            elbow = np.array((x + side * 0.012, torso_y + 0.015, waist_z + 0.03))
            wrist = np.array((mouth[0] + side * 0.10, torso_y - 0.25, waist_z + 0.015))
            self._limb(f"upper{name}", shoulder, elbow, 0.045, shirt)
            self._limb(f"fore{name}", elbow, wrist, 0.035, shirt)
            self.ball(f"hand{name}", skin, wrist + (0, -0.03, 0), (0.033, 0.046, 0.016))
            hip = np.array((mouth[0] + side * 0.095, torso_y, waist_z - 0.03))
            knee = np.array((mouth[0] + side * 0.10, torso_y - 0.40, waist_z - 0.03))
            ankle = np.array((mouth[0] + side * 0.10, torso_y - 0.42, FLOOR_Z + 0.09))
            self._limb(f"thigh{name}", hip, knee, 0.075, trousers)
            self._limb(f"shin{name}", knee, ankle, 0.055, trousers)
            shoe = ankle + (0, -0.05, -0.055)
            self.ball(f"shoe{name}", (0.14, 0.13, 0.13), shoe, (0.05, 0.12, 0.04))
        self._build_chair(mouth[0], torso_y, waist_z)

    def _limb(self, name, start, end, radius, color) -> None:
        """Place a tapered, round-ended limb between two points."""
        direction = end - start
        length = float(np.linalg.norm(direction))
        axis = np.cross((0.0, 0.0, 1.0), direction / length)
        angle = math.acos(float(np.clip(direction[2] / length, -1, 1)))
        if np.linalg.norm(axis) < 1e-6:
            orientation = p.getQuaternionFromAxisAngle((1, 0, 0), angle)
        else:
            orientation = p.getQuaternionFromAxisAngle(axis.tolist(), angle)
        arc = [i * math.pi / 8 for i in range(-4, 5)]
        cap_low = [(radius * math.cos(a), radius * math.sin(a)) for a in arc[:5]]
        cap_high = [
            (radius * 0.85 * math.cos(a), length + radius * 0.85 * math.sin(a)) for a in arc[4:]
        ]
        points = [(max(r, 1e-4), z) for r, z in cap_low + cap_high]
        self.place(name, lathe(points), color, start, orientation=orientation)

    def _build_chair(self, x: float, torso_y: float, waist_z: float) -> None:
        color = (0.36, 0.25, 0.16)
        seat_z = waist_z - 0.115
        seat_y = torso_y - 0.10
        self.place("seat", box((0.46, 0.46, 0.04)), color, (x, seat_y, seat_z))
        back = box((0.46, 0.03, 0.52))
        self.place("chair_back", back, color, (x, torso_y + 0.14, seat_z + 0.26))
        height = seat_z - FLOOR_Z
        for index, (dx, dy) in enumerate(((-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2))):
            position = (x + dx, seat_y + dy, FLOOR_Z + height / 2)
            self.place(f"chair_leg{index}", box((0.04, 0.04, height)), color, position)
