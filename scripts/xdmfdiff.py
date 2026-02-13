#!/usr/bin/env python3
"""Utility for reading and comparing XDMF outputs (including parallel multi-grid layouts)."""

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

def _parse_int_tuple(text: Optional[str]) -> Tuple[int, ...]:
    """Parse a whitespace-separated dimension string into a tuple of ints."""
    if not text:
        raise RuntimeError("Missing integer dimensions in XDMF file.")
    return tuple(int(v) for v in text.split())

def _read_numeric_text(element: ET.Element, dtype=float) -> np.ndarray:
    """Read numeric values from an XML DataItem element."""
    text = ''.join(element.itertext()).strip()
    if not text:
        raise RuntimeError("Empty DataItem encountered while parsing XDMF.")
    return np.fromstring(text, sep=' ', dtype=dtype)

def _normalize_center(value: Optional[str]) -> Optional[str]:
    """Normalize XDMF center labels to 'Cell' or 'Node' (or return None)."""
    center = (value or '').strip().lower()
    if center in ('node', 'nodes', 'point', 'points'):
        return 'Node'
    if center in ('cell', 'cells', 'element', 'elements'):
        return 'Cell'
    return None

def _xdmf_datatype(array: np.ndarray) -> str:
    """Map numpy dtype to XDMF DataType strings."""
    if np.issubdtype(array.dtype, np.integer):
        return 'Int'
    if np.issubdtype(array.dtype, np.bool_):
        return 'Int'
    return 'Float'

def _format_dims(shape: Sequence[int]) -> str:
    """Format dimensions for XDMF output attributes."""
    return ' '.join(str(v) for v in shape)

def _sanitize_name(name: str) -> str:
    """Return a filesystem-safe field name."""
    return re.sub(r'[^0-9A-Za-z_.-]', '_', name)

class HDFDataStore:
    """Cache HDF5 file handles and provide dataset reads."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._handles: Dict[str, h5py.File] = {}

    def read(self, spec: str) -> np.ndarray:
        if ':' not in spec:
            raise RuntimeError(f"Malformed HDF DataItem reference '{spec}'.")
        file_part, dataset_part = spec.split(':', 1)
        file_part = file_part.strip()
        dataset_part = dataset_part.strip()
        if not dataset_part.startswith('/'):
            dataset_part = '/' + dataset_part
        file_path = file_part if os.path.isabs(file_part) else os.path.join(self.base_dir, file_part)
        if file_path not in self._handles:
            if not os.path.exists(file_path):
                raise RuntimeError(f"Referenced HDF5 file '{file_path}' was not found.")
            self._handles[file_path] = h5py.File(file_path, 'r')
        dataset = self._handles[file_path][dataset_part]
        return np.array(dataset)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

@dataclass
class MeshInfo:
    """Global mesh metadata extracted from the XDMF Domain."""
    topology_type: str
    geometry_type: str
    node_dims: Tuple[int, ...]
    origin: np.ndarray
    spacing: np.ndarray

    @property
    def ndim(self) -> int:
        return len(self.node_dims)

    @property
    def node_shape(self) -> Tuple[int, ...]:
        return self.node_dims

    @property
    def cell_shape(self) -> Tuple[int, ...]:
        return tuple(max(d - 1, 0) for d in self.node_dims)

@dataclass
class AttributeData:
    """Concrete attribute payload loaded into memory."""
    name: str
    center: str
    values: np.ndarray

@dataclass
class DataItemSpec:
    """Deferred DataItem reference for lazy dataset loading."""
    fmt: str
    text: str
    dims: Optional[Tuple[int, ...]]
    dtype: Optional[str]

@dataclass
class AttributeSpec:
    """Deferred attribute metadata for lazy dataset loading."""
    name: str
    center: str
    data_item: DataItemSpec

@dataclass
class UniformGridSpec:
    """Deferred uniform grid metadata for lazy dataset loading."""
    name: str
    path: str
    node_dims: Tuple[int, ...]
    origin: np.ndarray
    spacing: np.ndarray
    attributes: List[AttributeSpec]

@dataclass
class StepSpec:
    """Collection of grids that belong to a single timestep."""
    step_id: str
    index: int
    time_value: Optional[float]
    grids: List[UniformGridSpec] = field(default_factory=list)

@dataclass
class UniformGridData:
    """Concrete uniform grid data with loaded attributes."""
    name: str
    path: str
    node_dims: Tuple[int, ...]
    origin: np.ndarray
    spacing: np.ndarray
    attributes: List[AttributeData]

    def spatial_shape(self, center: str) -> Tuple[int, ...]:
        if center == 'Node':
            return self.node_dims
        if center == 'Cell':
            return tuple(max(d - 1, 0) for d in self.node_dims)
        raise RuntimeError(f"Unsupported attribute center '{center}'.")

class FieldAccumulator:
    """Accumulate per-rank grid blocks into a single global array."""
    def __init__(self, mesh_shape: Tuple[int, ...], component_shape: Tuple[int, ...], dtype: np.dtype, ndim: int):
        self.component_shape = component_shape
        self.ndim = ndim
        shape = mesh_shape + component_shape
        self.data = np.zeros(shape, dtype=dtype)
        self.mask = np.zeros(mesh_shape, dtype=bool)

    def insert(self, offset: Tuple[int, ...], block: np.ndarray, field_name: str, grid_name: str) -> None:
        spatial_shape = block.shape[:self.ndim]
        comp_shape = block.shape[self.ndim:]
        if comp_shape != self.component_shape:
            raise RuntimeError(
                f"Component shape mismatch for field '{field_name}' on grid '{grid_name}'."
            )
        slices = tuple(slice(offset[i], offset[i] + spatial_shape[i]) for i in range(self.ndim))
        if self.mask[slices].any():
            raise RuntimeError(f"Overlapping data detected for field '{field_name}' on grid '{grid_name}'.")
        self.data[slices + (slice(None),) * len(comp_shape)] = block
        self.mask[slices] = True

@dataclass
class Snapshot:
    """Merged field data for a single timestep."""
    step_id: str
    index: int
    time_value: Optional[float]
    cell_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    node_fields: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.time_value is not None:
            return f"t={self.time_value:g}"
        return f"step={self.index}"

class SnapshotBuilder:
    """Build a Snapshot by inserting per-grid blocks at computed offsets."""
    def __init__(self, mesh: MeshInfo, step_id: str, index: int, time_value: Optional[float]):
        self.mesh = mesh
        self.step_id = step_id
        self.index = index
        self.time_value = time_value
        self.accumulators: Dict[str, Dict[str, FieldAccumulator]] = {'Cell': {}, 'Node': {}}

    def ensure_time_value(self, time_value: Optional[float]) -> None:
        if self.time_value is None and time_value is not None:
            self.time_value = time_value

    def add_grid(self, grid_data: UniformGridData, offset: Tuple[int, ...]) -> None:
        for attr in grid_data.attributes:
            center = attr.center
            if center not in self.accumulators:
                continue
            block = attr.values
            spatial_shape = grid_data.spatial_shape(center)
            if block.shape[: self.mesh.ndim] != spatial_shape:
                raise RuntimeError(
                    f"Attribute '{attr.name}' on grid '{grid_data.path}' has spatial shape {block.shape[: self.mesh.ndim]} "
                    f"but expected {spatial_shape}."
                )
            accumulator = self._get_accumulator(center, attr.name, block)
            accumulator.insert(offset, block, attr.name, grid_data.path)

    def _get_accumulator(self, center: str, name: str, block: np.ndarray) -> FieldAccumulator:
        accum_map = self.accumulators[center]
        component_shape = block.shape[self.mesh.ndim :]
        dtype = block.dtype
        mesh_shape = self.mesh.cell_shape if center == 'Cell' else self.mesh.node_shape
        accumulator = accum_map.get(name)
        if accumulator is None:
            accumulator = FieldAccumulator(mesh_shape, component_shape, dtype, self.mesh.ndim)
            accum_map[name] = accumulator
        else:
            if accumulator.component_shape != component_shape:
                raise RuntimeError(
                    f"Component shape mismatch while merging field '{name}' in step '{self.step_id}'."
                )
            if accumulator.data.dtype != block.dtype:
                accumulator.data = accumulator.data.astype(np.result_type(accumulator.data.dtype, block.dtype))
        return accumulator

    def build(self) -> Snapshot:
        cell_fields = {}
        node_fields = {}
        for center, accum_map in self.accumulators.items():
            for name, accumulator in accum_map.items():
                if not accumulator.mask.all():
                    missing = accumulator.mask.size - int(accumulator.mask.sum())
                    raise RuntimeError(
                        f"Incomplete coverage for {center.lower()} field '{name}' in step '{self.step_id}'. Missing {missing} entries."
                    )
                target = cell_fields if center == 'Cell' else node_fields
                target[name] = accumulator.data
        return Snapshot(step_id=self.step_id, index=self.index, time_value=self.time_value, cell_fields=cell_fields, node_fields=node_fields)

class OffsetResolver:
    """Compute grid offsets from origin/spacing and validate coverage."""
    def __init__(self, mesh: MeshInfo):
        self.mesh = mesh
        self._cache: Dict[Tuple[str, Tuple[int, ...]], Tuple[int, ...]] = {}

    def resolve(self, grid_name: str, node_dims: Tuple[int, ...], origin: np.ndarray) -> Tuple[int, ...]:
        key = (grid_name, node_dims)
        offset = self._compute_offset(origin)
        if offset is None:
            if key in self._cache:
                print(
                    f"Warning: geometry for grid '{grid_name}' is inconsistent; reusing cached offset.",
                    file=sys.stderr,
                )
                return self._cache[key]
            raise RuntimeError(f"Unable to determine offset for grid '{grid_name}'.")
        if key in self._cache and self._cache[key] != offset:
            print(
                f"Warning: geometry for grid '{grid_name}' changed between steps; reusing cached offset.",
                file=sys.stderr,
            )
            return self._cache[key]
        if not self._fits_within(offset, node_dims):
            if key in self._cache:
                print(
                    f"Warning: computed offset for grid '{grid_name}' is out of bounds; reusing cached offset.",
                    file=sys.stderr,
                )
                return self._cache[key]
            raise RuntimeError(f"Grid '{grid_name}' does not fit within the global mesh extents.")
        self._cache[key] = offset
        return offset

    def _compute_offset(self, origin: np.ndarray) -> Optional[Tuple[int, ...]]:
        offsets: List[int] = []
        for ori, base, spacing in zip(origin, self.mesh.origin, self.mesh.spacing):
            if math.isclose(spacing, 0.0, abs_tol=1e-15):
                if not math.isclose(ori, base, rel_tol=0.0, abs_tol=1e-7):
                    return None
                offsets.append(0)
                continue
            rel = (ori - base) / spacing
            rounded = int(round(rel))
            if not math.isclose(rel, rounded, rel_tol=0.0, abs_tol=1e-4):
                return None
            offsets.append(rounded)
        offsets_array = np.array(offsets, dtype=float)
        reconstructed = self.mesh.origin + offsets_array * self.mesh.spacing
        tol = np.maximum(1e-6, np.abs(self.mesh.spacing) * 1e-4)
        if not np.allclose(reconstructed, origin, rtol=0.0, atol=tol):
            return None
        return tuple(offsets)

    def _fits_within(self, offset: Tuple[int, ...], node_dims: Tuple[int, ...]) -> bool:
        for off, size, global_size in zip(offset, node_dims, self.mesh.node_shape):
            if off < 0 or off + size > global_size:
                return False
        return True

@dataclass
class TraverseContext:
    time_value: Optional[float]
    step_id: Optional[str]

class XdmfSeries:
    """Parse an XDMF series and load timesteps on demand."""
    def __init__(self, path: str):
        self.path = path
        self.base_dir = os.path.dirname(os.path.abspath(path))
        tree = ET.parse(path)
        self.root = tree.getroot()
        domain = self.root.find('Domain')
        if domain is None:
            raise RuntimeError(f"File '{path}' does not contain an Xdmf Domain element.")
        self.domain = domain
        self.mesh = self._parse_mesh()
        self._hdf_store = HDFDataStore(self.base_dir)
        self._offset_resolver = OffsetResolver(self.mesh)
        self.steps = self._collect_steps()

    def close(self) -> None:
        """Close any cached HDF5 handles."""
        self._hdf_store.close()

    def _parse_mesh(self) -> MeshInfo:
        """Parse global mesh topology and geometry."""
        topology = self.domain.find('Topology')
        geometry = self.domain.find('Geometry')
        if topology is None or geometry is None:
            raise RuntimeError("Domain must define Topology and Geometry blocks.")
        topology_type = topology.attrib.get('TopologyType', '')
        if 'corectmesh' not in topology_type.lower():
            raise RuntimeError("Only CoRectMesh topologies are supported by this script.")
        node_dims = _parse_int_tuple(topology.attrib.get('Dimensions'))
        geometry_type = geometry.attrib.get('Type', '')
        origin, spacing = self._parse_geometry(geometry)
        if len(node_dims) != origin.size or origin.size != spacing.size:
            raise RuntimeError("Mismatch between topology dimensions and geometry vectors.")
        return MeshInfo(
            topology_type=topology_type,
            geometry_type=geometry_type,
            node_dims=node_dims,
            origin=origin,
            spacing=spacing,
        )

    def _parse_geometry(self, geometry: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
        """Parse origin and spacing from a Geometry block."""
        geom_type = (geometry.attrib.get('Type', '') or '').upper()
        data_items = geometry.findall('DataItem')
        if not geom_type.startswith('ORIGIN_DX') or len(data_items) < 2:
            raise RuntimeError(f"Unsupported geometry definition '{geom_type}'.")
        origin = _read_numeric_text(data_items[0])
        spacing = _read_numeric_text(data_items[1])
        return origin.astype(float), spacing.astype(float)

    def _read_data_array(self, spec: DataItemSpec) -> np.ndarray:
        """Read a DataItem payload into a numpy array."""
        fmt = spec.fmt.upper()
        if fmt == 'HDF':
            data = self._hdf_store.read(spec.text)
        elif fmt == 'XML':
            dtype_attr = (spec.dtype or '').strip().lower()
            dtype = int if dtype_attr in ('int', 'integer') else float
            data = np.fromstring(spec.text, sep=' ', dtype=dtype)
        else:
            raise RuntimeError(f"Unsupported DataItem format '{spec.fmt}'.")
        if spec.dims:
            if data.size != int(np.prod(spec.dims)):
                raise RuntimeError(
                    f"DataItem declared dimensions {spec.dims} but contains {data.size} values."
                )
            data = data.reshape(spec.dims, order='C')
        return data

    def _parse_data_item(self, data_item: ET.Element) -> DataItemSpec:
        """Parse a DataItem element into a deferred spec."""
        fmt = (data_item.attrib.get('Format', 'XML') or '').strip().upper()
        text = ''.join(data_item.itertext()).strip()
        dims_text = data_item.attrib.get('Dimensions')
        dims = _parse_int_tuple(dims_text) if dims_text else None
        dtype = data_item.attrib.get('DataType')
        return DataItemSpec(fmt=fmt, text=text, dims=dims, dtype=dtype)

    def _parse_uniform_grid(self, grid: ET.Element, path: str) -> UniformGridSpec:
        """Parse a uniform grid definition into a deferred spec."""
        topology = grid.find('Topology')
        if topology is None:
            topology = self.domain.find('Topology')
        geometry = grid.find('Geometry')
        if geometry is None:
            geometry = self.domain.find('Geometry')
        if topology is None or geometry is None:
            raise RuntimeError(f"Grid '{path}' is missing topology or geometry definitions.")
        node_dims = _parse_int_tuple(topology.attrib.get('Dimensions'))
        geom_origin, geom_spacing = self._parse_geometry(geometry)
        attributes: List[AttributeSpec] = []
        for attr in grid.findall('Attribute'):
            center = _normalize_center(attr.attrib.get('Center'))
            if center is None:
                print(
                    f"Warning: skipping attribute '{attr.attrib.get('Name', 'unnamed')}' with unsupported center '{attr.attrib.get('Center')}'.",
                    file=sys.stderr,
                )
                continue
            data_item = attr.find('DataItem')
            if data_item is None:
                raise RuntimeError(f"Attribute '{attr.attrib.get('Name')}' in grid '{path}' has no DataItem.")
            attributes.append(
                AttributeSpec(
                    name=attr.attrib.get('Name', 'unnamed'),
                    center=center,
                    data_item=self._parse_data_item(data_item),
                )
            )
        return UniformGridSpec(
            name=grid.attrib.get('Name', path.split('/')[-1]),
            path=path,
            node_dims=node_dims,
            origin=geom_origin,
            spacing=geom_spacing,
            attributes=attributes,
        )

    def _collect_steps(self) -> List[StepSpec]:
        """Collect grids grouped by timestep without loading data."""
        steps: Dict[str, StepSpec] = {}
        order: List[str] = []

        def get_step(step_id: str, time_value: Optional[float]) -> StepSpec:
            if step_id not in steps:
                step = StepSpec(step_id=step_id, index=len(order), time_value=time_value)
                steps[step_id] = step
                order.append(step_id)
            step = steps[step_id]
            if step.time_value is None and time_value is not None:
                step.time_value = time_value
            return step

        def walk(grid: ET.Element, context: TraverseContext, name_stack: List[str]) -> None:
            grid_name = grid.attrib.get('Name', f"Grid{len(name_stack)}")
            current_path_items = name_stack + [grid_name]
            current_path = '/'.join(current_path_items)
            time_value = context.time_value
            step_id = context.step_id
            time_elem = grid.find('Time')
            if time_elem is not None and 'Value' in time_elem.attrib:
                try:
                    time_value = float(time_elem.attrib['Value'])
                except ValueError as exc:  # pragma: no cover - malformed files
                    raise RuntimeError(
                        f"Invalid time value '{time_elem.attrib['Value']}' in grid '{grid_name}'."
                    ) from exc
                step_id = f"time:{time_value}"
            elif step_id is None:
                step_id = current_path
            grid_type = (grid.attrib.get('GridType', 'Uniform') or '').strip().lower()
            if grid_type == 'collection':
                child_context = TraverseContext(time_value=time_value, step_id=step_id)
                for child in grid.findall('Grid'):
                    walk(child, child_context, current_path_items)
            else:
                step = get_step(step_id, time_value)
                uniform = self._parse_uniform_grid(grid, current_path)
                if len(uniform.node_dims) != self.mesh.ndim:
                    raise RuntimeError(f"Grid '{current_path}' dimensionality mismatch.")
                step.grids.append(uniform)

        for top_grid in self.domain.findall('Grid'):
            walk(top_grid, TraverseContext(time_value=None, step_id=None), [])
        return [steps[key] for key in order]

    def list_steps(self) -> List[Tuple[int, Optional[float], str]]:
        """Return (index, time_value, step_id) for each timestep."""
        return [(step.index, step.time_value, step.step_id) for step in self.steps]

    def _find_step(self,
                  step_index: Optional[int] = None,
                  time_value: Optional[float] = None,
                  step_id: Optional[str] = None) -> StepSpec:
        """Locate a StepSpec without loading data."""
        if step_id is not None:
            for candidate in self.steps:
                if candidate.step_id == step_id:
                    return candidate
            raise RuntimeError(f"Step id '{step_id}' not found in XDMF.")
        if step_index is not None:
            if step_index < 0 or step_index >= len(self.steps):
                raise RuntimeError(f"Step index {step_index} is out of range.")
            return self.steps[step_index]
        if time_value is not None:
            for candidate in self.steps:
                if candidate.time_value is not None and math.isclose(candidate.time_value, time_value, rel_tol=1e-9, abs_tol=1e-12):
                    return candidate
            raise RuntimeError(f"No step found at time {time_value}.")
        raise RuntimeError("Must provide step_index, time_value, or step_id.")


    @property
    def has_explicit_times(self) -> bool:
        return all(step.time_value is not None for step in self.steps)

    def load_snapshot(self,
                      step_index: Optional[int] = None,
                      time_value: Optional[float] = None,
                      step_id: Optional[str] = None,
                      centers: Sequence[str] = ('Cell', 'Node')) -> Snapshot:
        """Load a single timestep into a merged Snapshot."""
        step = None
        if step_id is not None:
            for candidate in self.steps:
                if candidate.step_id == step_id:
                    step = candidate
                    break
        elif step_index is not None:
            if step_index < 0 or step_index >= len(self.steps):
                raise RuntimeError(f"Step index {step_index} is out of range.")
            step = self.steps[step_index]
        elif time_value is not None:
            for candidate in self.steps:
                if candidate.time_value is not None and math.isclose(candidate.time_value, time_value, rel_tol=1e-9, abs_tol=1e-12):
                    step = candidate
                    break
        else:
            raise RuntimeError("Must provide step_index, time_value, or step_id.")

        if step is None:
            raise RuntimeError("Unable to locate requested timestep in XDMF.")

        builder = SnapshotBuilder(self.mesh, step.step_id, step.index, step.time_value)
        for grid in step.grids:
            attributes: List[AttributeData] = []
            for attr in grid.attributes:
                if attr.center not in centers:
                    continue
                values = self._read_data_array(attr.data_item)
                attributes.append(AttributeData(name=attr.name, center=attr.center, values=values))
            if not attributes:
                continue
            uniform = UniformGridData(
                name=grid.name,
                path=grid.path,
                node_dims=grid.node_dims,
                origin=grid.origin,
                spacing=grid.spacing,
                attributes=attributes,
            )
            offset = self._offset_resolver.resolve(uniform.name, uniform.node_dims, uniform.origin)
            builder.add_grid(uniform, offset)
        return builder.build()

def _ensure_mesh_compatibility(mesh_a: MeshInfo, mesh_b: MeshInfo) -> None:
    """Validate that two meshes have matching topology and geometry."""
    if mesh_a.topology_type != mesh_b.topology_type:
        raise RuntimeError("Topology types differ between the two files.")
    if mesh_a.node_shape != mesh_b.node_shape:
        raise RuntimeError("Topology dimensions differ between the two files.")
    if not np.allclose(mesh_a.origin, mesh_b.origin):
        raise RuntimeError("Origin vectors differ between the two files.")
    if not np.allclose(mesh_a.spacing, mesh_b.spacing):
        raise RuntimeError("Grid spacing differs between the two files.")

def _align_steps(series_a: XdmfSeries, series_b: XdmfSeries) -> List[Tuple[StepSpec, StepSpec]]:
    """Match timestep ordering between two series (by time when present)."""
    if len(series_a.steps) != len(series_b.steps):
        raise RuntimeError("The two files contain a different number of time steps.")
    if series_a.has_explicit_times and series_b.has_explicit_times:
        pairs: List[Tuple[StepSpec, StepSpec]] = []
        used: set[int] = set()
        for step_a in series_a.steps:
            match_idx = None
            for idx, step_b in enumerate(series_b.steps):
                if idx in used:
                    continue
                if step_a.time_value is None or step_b.time_value is None:
                    continue
                if math.isclose(step_a.time_value, step_b.time_value, rel_tol=1e-9, abs_tol=1e-12):
                    match_idx = idx
                    break
            if match_idx is None:
                raise RuntimeError(f"No matching time value found for step at {step_a.time_value}.")
            used.add(match_idx)
            pairs.append((step_a, series_b.steps[match_idx]))
        return pairs
    if series_a.has_explicit_times != series_b.has_explicit_times:
        raise RuntimeError("Only one file defines explicit times; cannot align snapshots.")
    return list(zip(series_a.steps, series_b.steps))

def _collect_attr_centers(step: StepSpec) -> Dict[str, str]:
    """Collect attribute centers by name; error if inconsistent within a file."""
    centers: Dict[str, str] = {}
    for grid in step.grids:
        for attr in grid.attributes:
            name = attr.name
            center = attr.center
            if name in centers and centers[name] != center:
                raise RuntimeError(
                    f"{step.step_id}: attribute '{name}' appears with multiple centers ({centers[name]} vs {center})."
                )
            centers[name] = center
    return centers

def _ensure_step_centers_match(step_a: StepSpec, step_b: StepSpec) -> None:
    """Ensure both files use the same centers for shared attributes and overall presence."""
    centers_a = _collect_attr_centers(step_a)
    centers_b = _collect_attr_centers(step_b)

    for name in sorted(set(centers_a.keys()) & set(centers_b.keys())):
        if centers_a[name] != centers_b[name]:
            raise RuntimeError(
                f"{step_a.step_id}: center mismatch for '{name}' (fileA {centers_a[name]} vs fileB {centers_b[name]})."
            )

    has_cell_a = any(c == 'Cell' for c in centers_a.values())
    has_cell_b = any(c == 'Cell' for c in centers_b.values())
    if has_cell_a != has_cell_b:
        raise RuntimeError(
            f"{step_a.step_id}: center mismatch for 'Cell' (fileA has={has_cell_a}, fileB has={has_cell_b})."
        )

    has_node_a = any(c == 'Node' for c in centers_a.values())
    has_node_b = any(c == 'Node' for c in centers_b.values())
    if has_node_a != has_node_b:
        raise RuntimeError(
            f"{step_a.step_id}: center mismatch for 'Node' (fileA has={has_node_a}, fileB has={has_node_b})."
        )

def _compare_fields(time_label: str,
                    center_label: str,
                    fields_a: Dict[str, np.ndarray],
                    fields_b: Dict[str, np.ndarray],
                    abs_error: float,
                    rel_error: float,
                    abs_zero: float) -> Dict[str, np.ndarray]:
    """Compute diffs and report only when abs/rel thresholds are exceeded."""
    diffs: Dict[str, np.ndarray] = {}
    common = sorted(set(fields_a.keys()) & set(fields_b.keys()))
    if not common:
        print(f"{time_label}: no common {center_label} fields to compare.")
        return diffs
    for name in common:
        a = fields_a[name]
        b = fields_b[name]
        if a.shape != b.shape:
            print(f"{time_label}: skipping {center_label} field '{name}' (shape mismatch {a.shape} vs {b.shape}).")
            continue
        diff = a - b
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        ref_max = float(np.max(np.abs(a))) if a.size else 0.0
        denom = max(ref_max, abs_zero)
        rel = max_abs / denom if denom > 0.0 else (0.0 if max_abs == 0.0 else float('inf'))
        if max_abs > abs_error or rel > rel_error:
            print(
                f"{time_label}: {center_label} field '{name}': "
                f"max_abs={max_abs:.6e}, rel={rel:.6e}, ref_max={ref_max:.6e}"
            )
            diffs[name] = diff
    return diffs

def _write_diff_series(output_path: str, mesh: MeshInfo, snapshots: List[Snapshot]) -> None:
    """Write a diff XDMF/HDF5 series from per-step difference fields."""
    snapshots_with_data = [snap for snap in snapshots if snap.cell_fields or snap.node_fields]
    if not snapshots_with_data:
        print("No difference fields available to write.")
        return
    xmf_path = Path(output_path)
    h5_path = xmf_path.with_suffix('.h5')
    dataset_map: Dict[Tuple[int, str, str], str] = {}
    with h5py.File(h5_path, 'w') as handle:
        for step_idx, snap in enumerate(snapshots_with_data):
            for center_label, fields in (('Node', snap.node_fields), ('Cell', snap.cell_fields)):
                for name, data in fields.items():
                    dataset_name = f"{_sanitize_name(name)}_{center_label.lower()}_{step_idx}"
                    handle.create_dataset(dataset_name, data=data)
                    dataset_map[(step_idx, center_label, name)] = dataset_name
    root = ET.Element('Xdmf', attrib={'Version': '3.0', 'xmlns:xi': 'http://www.w3.org/2003/XInclude'})
    domain = ET.SubElement(root, 'Domain')
    ET.SubElement(domain, 'Topology', TopologyType=mesh.topology_type, Dimensions=_format_dims(mesh.node_shape))
    geom = ET.SubElement(domain, 'Geometry', Type=mesh.geometry_type)
    origin_item = ET.SubElement(geom, 'DataItem', Format='XML', Dimensions=str(mesh.ndim))
    origin_item.text = ' '.join(f"{val:.16g}" for val in mesh.origin)
    spacing_item = ET.SubElement(geom, 'DataItem', Format='XML', Dimensions=str(mesh.ndim))
    spacing_item.text = ' '.join(f"{val:.16g}" for val in mesh.spacing)
    ts_grid = ET.SubElement(domain, 'Grid', Name='TimeSeries', GridType='Collection', CollectionType='Temporal')
    h5_name = h5_path.name
    for step_idx, snap in enumerate(snapshots_with_data):
        grid = ET.SubElement(ts_grid, 'Grid', Name=f"T{step_idx}", GridType='Uniform')
        time_value = snap.time_value if snap.time_value is not None else step_idx
        ET.SubElement(grid, 'Time', Value=f"{time_value}")
        ET.SubElement(grid, 'Topology', TopologyType=mesh.topology_type, Dimensions=_format_dims(mesh.node_shape))
        grid_geom = ET.SubElement(grid, 'Geometry', Type=mesh.geometry_type)
        g_origin = ET.SubElement(grid_geom, 'DataItem', Format='XML', Dimensions=str(mesh.ndim))
        g_origin.text = origin_item.text
        g_spacing = ET.SubElement(grid_geom, 'DataItem', Format='XML', Dimensions=str(mesh.ndim))
        g_spacing.text = spacing_item.text
        for center_label, fields in (('Node', snap.node_fields), ('Cell', snap.cell_fields)):
            for name, data in fields.items():
                attr = ET.SubElement(grid, 'Attribute', Name=name, Center=center_label)
                dims = _format_dims(data.shape)
                data_item = ET.SubElement(
                    attr,
                    'DataItem',
                    DataType=_xdmf_datatype(data),
                    Dimensions=dims,
                    Format='HDF',
                )
                dataset_name = dataset_map[(step_idx, center_label, name)]
                data_item.text = f"{h5_name}:/{dataset_name}"
    tree = ET.ElementTree(root)
    try:  # pragma: no cover - ElementTree.indent introduced in Python 3.9
        ET.indent(tree, space='  ')
    except AttributeError:
        pass
    tree.write(xmf_path, encoding='utf-8', xml_declaration=True)
    print(f"Wrote difference series to {xmf_path} (with data in {h5_path}).")

def compare_series(path_a: str,
                   path_b: str,
                   diff_out: Optional[str],
                   step_index: Optional[int] = None,
                   time_value: Optional[float] = None,
                   centers: Sequence[str] = ('Cell',),
                   abs_error: float = 0.0,
                   rel_error: float = 0.0,
                   abs_zero: float = 0.0) -> bool:
    """Compare two XDMF series and optionally emit a diff file."""
    series_a = XdmfSeries(path_a)
    series_b = XdmfSeries(path_b)
    try:
        _ensure_mesh_compatibility(series_a.mesh, series_b.mesh)
        print('Mesh topology matches - safe to compare fields.')
        diff_snapshots: List[Snapshot] = []
        if step_index is not None or time_value is not None:
            step_a = series_a._find_step(step_index=step_index, time_value=time_value)
            step_b = series_b._find_step(step_index=step_index, time_value=time_value)
            _ensure_step_centers_match(step_a, step_b)
            snap_a = series_a.load_snapshot(step_index=step_a.index, time_value=step_a.time_value, centers=centers)
            snap_b = series_b.load_snapshot(step_index=step_b.index, time_value=step_b.time_value, centers=centers)
            label = snap_a.label
            diff_snapshot = Snapshot(step_id=snap_a.step_id, index=snap_a.index, time_value=snap_a.time_value)
            node_diff = _compare_fields(label, 'point', snap_a.node_fields, snap_b.node_fields, abs_error, rel_error, abs_zero)
            cell_diff = _compare_fields(label, 'cell', snap_a.cell_fields, snap_b.cell_fields, abs_error, rel_error, abs_zero)
            diff_snapshot.node_fields = node_diff
            diff_snapshot.cell_fields = cell_diff
            if node_diff or cell_diff:
                diff_snapshots.append(diff_snapshot)
        else:
            for step_a, step_b in _align_steps(series_a, series_b):
                _ensure_step_centers_match(step_a, step_b)
                snap_a = series_a.load_snapshot(step_index=step_a.index, centers=centers)
                snap_b = series_b.load_snapshot(step_index=step_b.index, centers=centers)
                label = snap_a.label
                diff_snapshot = Snapshot(step_id=snap_a.step_id, index=snap_a.index, time_value=snap_a.time_value)
                node_diff = _compare_fields(label, 'point', snap_a.node_fields, snap_b.node_fields, abs_error, rel_error, abs_zero)
                cell_diff = _compare_fields(label, 'cell', snap_a.cell_fields, snap_b.cell_fields, abs_error, rel_error, abs_zero)
                diff_snapshot.node_fields = node_diff
                diff_snapshot.cell_fields = cell_diff
                if node_diff or cell_diff:
                    diff_snapshots.append(diff_snapshot)
        if diff_out and diff_snapshots:
            _write_diff_series(diff_out, series_a.mesh, diff_snapshots)
        elif diff_out:
            print('No overlapping fields were found; skipping diff file export.')
        return bool(diff_snapshots)
    finally:
        series_a.close()
        series_b.close()

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI args for xdmfdiff."""
    parser = argparse.ArgumentParser(description='Compare fields stored in XDMF files.')
    parser.add_argument('file_a', help='Reference XDMF file (e.g., serial output).')
    parser.add_argument('file_b', help='XDMF file to compare (e.g., parallel output).')
    parser.add_argument('diff_out', nargs='?', help='Optional XDMF path for writing field differences.')
    parser.add_argument('--step', type=int, default=None, help='Compare only a single timestep by index.')
    parser.add_argument('--time', type=float, default=None, help='Compare only a single timestep by time value.')
    parser.add_argument('--centers',
                        choices=('cell', 'node', 'both'),
                        default='cell',
                        help='Which attribute centers to compare.')
    parser.add_argument('--abs-error', type=float, default=0.0, help='Absolute error threshold.')
    parser.add_argument('--rel-error', type=float, default=0.0, help='Relative error threshold.')
    parser.add_argument('--abs-zero',
                        type=float,
                        default=0.0,
                        help='Absolute floor used for relative error denominator.')
    return parser.parse_args(argv)

def main() -> None:
    """CLI entrypoint."""
    args = parse_args(sys.argv[1:])
    centers: Tuple[str, ...]
    if args.centers == 'both':
        centers = ('Cell', 'Node')
    elif args.centers == 'node':
        centers = ('Node',)
    else:
        centers = ('Cell',)
    try:
        has_diff = compare_series(args.file_a,
                                  args.file_b,
                                  args.diff_out,
                                  args.step,
                                  args.time,
                                  centers,
                                  args.abs_error,
                                  args.rel_error,
                                  args.abs_zero)
        if has_diff:
            sys.exit(1)
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
