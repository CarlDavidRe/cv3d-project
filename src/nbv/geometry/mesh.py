"""Deterministic loading of triangular ShapeNet meshes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class MeshError(ValueError):
    """Raised when a mesh cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    """A dependency-light triangular mesh in object/world coordinates."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    def __post_init__(self) -> None:
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise MeshError("vertices must have shape [N, 3]")
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise MeshError("faces must have shape [M, 3]")
        if len(vertices) == 0 or len(faces) == 0:
            raise MeshError("mesh must contain vertices and triangular faces")
        if not np.all(np.isfinite(vertices)):
            raise MeshError("mesh vertices must be finite")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise MeshError("mesh face index is outside the vertex array")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def triangles(self) -> NDArray[np.float64]:
        return self.vertices[self.faces]

    @property
    def bounds(self) -> NDArray[np.float64]:
        return np.stack((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    def scaled(self, scale: float) -> "TriangleMesh":
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and greater than zero")
        return TriangleMesh(self.vertices * scale, self.faces)


def load_obj_mesh(path: str | Path) -> TriangleMesh:
    """Load and fan-triangulate the geometry records in a ShapeNet OBJ file.

    Texture, normal, material, and group records are intentionally ignored.
    ShapeNetCore's ``models/model_normalized.obj`` geometry is represented by
    standard ``v`` and ``f`` records, including occasional polygonal faces.
    """

    source = Path(path)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    try:
        handle = source.open("r", encoding="utf-8", errors="strict")
    except FileNotFoundError as exc:
        raise MeshError(f"Mesh does not exist: {source}") from exc
    except OSError as exc:
        raise MeshError(f"Could not open mesh {source}: {exc}") from exc

    try:
        with handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.split("#", 1)[0].strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if fields[0] == "v":
                    if len(fields) < 4:
                        raise MeshError(
                            f"Invalid vertex at {source}:{line_number}"
                        )
                    vertices.append(tuple(float(value) for value in fields[1:4]))
                elif fields[0] == "f":
                    if len(fields) < 4:
                        raise MeshError(f"Invalid face at {source}:{line_number}")
                    polygon = [
                        _parse_obj_vertex_index(
                            token, len(vertices), source, line_number
                        )
                        for token in fields[1:]
                    ]
                    for index in range(1, len(polygon) - 1):
                        faces.append((polygon[0], polygon[index], polygon[index + 1]))
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, MeshError):
            raise
        raise MeshError(f"Invalid numeric value in OBJ mesh {source}") from exc

    return TriangleMesh(np.asarray(vertices), np.asarray(faces))


def load_ply_mesh(path: str | Path) -> TriangleMesh:
    """Load and fan-triangulate an ASCII PLY mesh.

    The ShapeNet subset used by this project stores geometry as ASCII
    ``model_normalized.ply`` files. Extra vertex properties and elements are
    accepted, but only vertex positions and face vertex indices are retained.
    """

    source = Path(path)
    try:
        handle = source.open("r", encoding="ascii", errors="strict")
    except FileNotFoundError as exc:
        raise MeshError(f"Mesh does not exist: {source}") from exc
    except OSError as exc:
        raise MeshError(f"Could not open mesh {source}: {exc}") from exc

    elements: list[dict[str, Any]] = []
    ply_format: str | None = None
    try:
        with handle:
            if handle.readline().strip() != "ply":
                raise MeshError(f"Invalid PLY signature in {source}")
            for line_number, raw_line in enumerate(handle, 2):
                fields = raw_line.strip().split()
                if not fields or fields[0] in {"comment", "obj_info"}:
                    continue
                if fields[0] == "format":
                    if len(fields) != 3:
                        raise MeshError(f"Invalid PLY format at {source}:{line_number}")
                    ply_format = fields[1]
                elif fields[0] == "element":
                    if len(fields) != 3:
                        raise MeshError(
                            f"Invalid PLY element at {source}:{line_number}"
                        )
                    count = int(fields[2])
                    if count < 0:
                        raise MeshError(
                            f"Negative PLY element count at {source}:{line_number}"
                        )
                    elements.append(
                        {"name": fields[1], "count": count, "properties": []}
                    )
                elif fields[0] == "property":
                    if not elements:
                        raise MeshError(
                            "PLY property precedes an element at "
                            f"{source}:{line_number}"
                        )
                    if len(fields) == 3:
                        prop = ("scalar", fields[1], fields[2])
                    elif len(fields) == 5 and fields[1] == "list":
                        prop = ("list", fields[2], fields[3], fields[4])
                    else:
                        raise MeshError(
                            f"Invalid PLY property at {source}:{line_number}"
                        )
                    elements[-1]["properties"].append(prop)
                elif fields[0] == "end_header":
                    break
            else:
                raise MeshError(f"PLY header has no end_header in {source}")

            if ply_format != "ascii":
                raise MeshError(
                    f"Unsupported PLY format {ply_format!r} in {source}; expected ascii"
                )

            vertices: list[tuple[float, float, float]] = []
            faces: list[tuple[int, int, int]] = []
            for element in elements:
                properties = element["properties"]
                for _ in range(element["count"]):
                    raw_line = handle.readline()
                    if not raw_line:
                        raise MeshError(
                            "Unexpected end of PLY data while reading "
                            f"{element['name']} in {source}"
                        )
                    values = _parse_ascii_ply_record(
                        raw_line.split(), properties, source
                    )
                    if element["name"] == "vertex":
                        try:
                            vertices.append(
                                (
                                    float(values["x"]),
                                    float(values["y"]),
                                    float(values["z"]),
                                )
                            )
                        except KeyError as exc:
                            raise MeshError(
                                f"PLY vertex element lacks x/y/z properties in {source}"
                            ) from exc
                    elif element["name"] == "face":
                        polygon_values = values.get(
                            "vertex_indices", values.get("vertex_index")
                        )
                        if polygon_values is None:
                            raise MeshError(
                                f"PLY face element lacks vertex_indices in {source}"
                            )
                        polygon = [int(value) for value in polygon_values]
                        if len(polygon) < 3:
                            raise MeshError(
                                f"PLY face has fewer than 3 vertices in {source}"
                            )
                        for index in range(1, len(polygon) - 1):
                            faces.append(
                                (polygon[0], polygon[index], polygon[index + 1])
                            )
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, MeshError):
            raise
        raise MeshError(f"Invalid numeric value in PLY mesh {source}") from exc

    return TriangleMesh(np.asarray(vertices), np.asarray(faces))


def load_mesh(path: str | Path) -> TriangleMesh:
    """Load a supported mesh based on its filename extension."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".obj":
        return load_obj_mesh(source)
    if suffix == ".ply":
        return load_ply_mesh(source)
    raise MeshError(f"Unsupported mesh extension {source.suffix!r}: {source}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_obj_vertex_index(
    token: str, vertex_count: int, source: Path, line_number: int
) -> int:
    raw_index = token.split("/", 1)[0]
    if not raw_index:
        raise MeshError(f"Missing face vertex at {source}:{line_number}")
    index = int(raw_index)
    if index == 0:
        raise MeshError(f"OBJ indices cannot be zero at {source}:{line_number}")
    resolved = index - 1 if index > 0 else vertex_count + index
    if resolved < 0 or resolved >= vertex_count:
        raise MeshError(f"Face index is out of range at {source}:{line_number}")
    return resolved


def _parse_ascii_ply_record(
    tokens: list[str], properties: list[tuple[str, ...]], source: Path
) -> dict[str, str | list[str]]:
    values: dict[str, str | list[str]] = {}
    cursor = 0
    for prop in properties:
        if prop[0] == "scalar":
            if cursor >= len(tokens):
                raise MeshError(f"Incomplete PLY record in {source}")
            values[prop[2]] = tokens[cursor]
            cursor += 1
        else:
            if cursor >= len(tokens):
                raise MeshError(f"Incomplete PLY list property in {source}")
            count = int(tokens[cursor])
            cursor += 1
            if count < 0 or cursor + count > len(tokens):
                raise MeshError(f"Invalid PLY list length in {source}")
            values[prop[3]] = tokens[cursor : cursor + count]
            cursor += count
    if cursor != len(tokens):
        raise MeshError(f"Unexpected values at end of PLY record in {source}")
    return values
