"""Split the requested STEP assembly into MuJoCo-ready binary STL meshes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct

import gmsh
import numpy as np


DEFAULT_STEP = Path("source_assembly.stp")
LINK_NAMES = ("LIANGAN1", "LIANGAN2", "LIANGAN3", "LIANGAN4")


def classify(name: str, bounds: tuple[float, ...]) -> str | None:
    side = "left" if (bounds[0] + bounds[3]) > 0 else "right"
    for link in LINK_NAMES:
        if name.endswith(f"/{link}/{link}"):
            return f"{side}_{link.lower()}"
    if "/电机-轮/" in name:
        return f"{side}_wheel"
    if "/轮腿组" not in name:
        return "chassis"
    return None


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    header = b"Generated directly from requested STEP assembly"
    with path.open("wb") as output:
        output.write(header[:80].ljust(80, b"\0"))
        output.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            edge1 = triangle[1] - triangle[0]
            edge2 = triangle[2] - triangle[0]
            normal = np.cross(edge1, edge2)
            length = np.linalg.norm(normal)
            if length:
                normal /= length
            output.write(struct.pack("<12fH", *normal, *triangle.reshape(-1), 0))


def write_mesh_parts(output_dir: Path, group: str, triangles: np.ndarray) -> list[Path]:
    max_faces = 190_000
    part_count = (len(triangles) + max_faces - 1) // max_faces
    outputs = []
    for index in range(part_count):
        suffix = f"_{index + 1}" if part_count > 1 else ""
        output = output_dir / f"{group}{suffix}.stl"
        write_binary_stl(output, triangles[index * max_faces : (index + 1) * max_faces])
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=Path, default=DEFAULT_STEP)
    parser.add_argument("--output", type=Path, default=Path("MJCF/step_meshes"))
    parser.add_argument("--mesh-size-mm", type=float, default=4.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.mesh_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size_mm)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.add("wheel_leg_step")
    gmsh.model.occ.importShapes(str(args.step), highestDimOnly=True)
    gmsh.model.occ.synchronize()

    groups: dict[str, list[int]] = {}
    for _, tag in gmsh.model.getEntities(3):
        name = gmsh.model.getEntityName(3, tag)
        group = classify(name, gmsh.model.getBoundingBox(3, tag))
        if group:
            groups.setdefault(group, []).append(tag)

    gmsh.model.mesh.generate(2)
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    nodes = {
        int(tag): np.asarray(point, dtype=np.float32)
        for tag, point in zip(node_tags, coordinates.reshape(-1, 3))
    }

    manifest = [
        f"source={args.step}",
        f"sha256={hashlib.sha256(args.step.read_bytes()).hexdigest()}",
        f"step_volumes={len(gmsh.model.getEntities(3))}",
        f"mesh_size_mm={args.mesh_size_mm}",
    ]
    for group, volumes in sorted(groups.items()):
        surfaces = {
            tag
            for _, tag in gmsh.model.getBoundary(
                [(3, tag) for tag in volumes], combined=False, oriented=False
            )
        }
        triangles = []
        for surface in surfaces:
            element_types, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
            for element_type, tags in zip(element_types, element_nodes):
                _, _, _, node_count, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                if node_count != 3:
                    continue
                for triangle_tags in tags.reshape(-1, 3):
                    points = np.asarray([nodes[int(tag)] for tag in triangle_tags])
                    # STEP: X=lateral, Y=forward, Z=up, millimetres.
                    triangles.append(points[:, [1, 0, 2]] * np.asarray([0.001, -0.001, 0.001]))
        outputs = write_mesh_parts(args.output, group, np.asarray(triangles, dtype=np.float32))
        manifest.append(
            f"{group}: volumes={len(volumes)} triangles={len(triangles)} "
            f"files={','.join(path.name for path in outputs)}"
        )
        print(manifest[-1])

    (args.output / "SOURCE.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    gmsh.finalize()


if __name__ == "__main__":
    main()
