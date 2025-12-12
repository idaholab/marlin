#!/usr/bin/env python3

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
    if not text:
        raise RuntimeError("Missing integer dimensions in XDMF file.")
    return tuple(int(v) for v in text.split())


def _read_numeric_text(element: ET.Element, dtype=float) -> np.ndarray:
    text = ''.join(element.itertext()).strip()
    if not text:
        raise RuntimeError("Empty DataItem encountered while parsing XDMF.")
    return np.fromstring(text, sep=' ', dtype=dtype)


def _normalize_center(value: Optional[str]) -> Optional[str]:
    center = (value or '').strip().lower()
    if center in ('node', 'nodes', 'point', 'points'):
        return 'Node'
    if center in ('cell', 'cells', 'element', 'elements'):
        return 'Cell'
    return None


def _xdmf_datatype(array: np.ndarray) -> str:
    if np.issubdtype(array.dtype, np.integer):
        return 'Int'
    if np.issubdtype(array.dtype, np.bool_):
        return 'Int'
    return 'Float'


def _format_dims(shape: Sequence[int]) -> str:
    return ' '.join(str(v) for v in shape)


def _sanitize_name(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z_.-]', '_', name)


class HDFDataStore:
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
    name: str
    center: str
    values: np.ndarray


@dataclass
class UniformGridData:
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
            if not math.isclose(rel, rounded, rel_tol=0.0, abs_tol=1e-5):
                return None
            offsets.append(rounded)
        offsets_array = np.array(offsets, dtype=float)
        reconstructed = self.mesh.origin + offsets_array * self.mesh.spacing
        if not np.allclose(reconstructed, origin, rtol=0.0, atol=1e-5):
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
        try:
            self.snapshots = self._collect_snapshots()
        finally:
            self._hdf_store.close()

    def _parse_mesh(self) -> MeshInfo:
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
        geom_type = (geometry.attrib.get('Type', '') or '').upper()
        data_items = geometry.findall('DataItem')
        if not geom_type.startswith('ORIGIN_DX') or len(data_items) < 2:
            raise RuntimeError(f"Unsupported geometry definition '{geom_type}'.")
        origin = _read_numeric_text(data_items[0])
        spacing = _read_numeric_text(data_items[1])
        return origin.astype(float), spacing.astype(float)

    def _read_data_array(self, data_item: ET.Element) -> np.ndarray:
        fmt = (data_item.attrib.get('Format', 'XML') or '').strip().upper()
        if fmt == 'HDF':
            spec = ''.join(data_item.itertext()).strip()
            return self._hdf_store.read(spec)
        if fmt == 'XML':
            dtype_attr = (data_item.attrib.get('DataType', '') or '').strip().lower()
            dtype = int if dtype_attr in ('int', 'integer') else float
            data = _read_numeric_text(data_item, dtype=dtype)
        else:
            raise RuntimeError(f"Unsupported DataItem format '{fmt}'.")
        dims_text = data_item.attrib.get('Dimensions')
        if dims_text:
            dims = _parse_int_tuple(dims_text)
            if data.size != int(np.prod(dims)):
                raise RuntimeError(
                    f"DataItem declared dimensions {dims} but contains {data.size} values."
                )
            data = data.reshape(dims, order='C')
        return data

    def _parse_uniform_grid(self, grid: ET.Element, path: str) -> UniformGridData:
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
        attributes: List[AttributeData] = []
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
            values = self._read_data_array(data_item)
            attributes.append(AttributeData(name=attr.attrib.get('Name', 'unnamed'), center=center, values=values))
        return UniformGridData(
            name=grid.attrib.get('Name', path.split('/')[-1]),
            path=path,
            node_dims=node_dims,
            origin=geom_origin,
            spacing=geom_spacing,
            attributes=attributes,
        )

    def _collect_snapshots(self) -> List[Snapshot]:
        builders: Dict[str, SnapshotBuilder] = {}
        order: List[str] = []

        def get_builder(step_id: str, time_value: Optional[float]) -> SnapshotBuilder:
            if step_id not in builders:
                builder = SnapshotBuilder(self.mesh, step_id, len(order), time_value)
                builders[step_id] = builder
                order.append(step_id)
            builder = builders[step_id]
            builder.ensure_time_value(time_value)
            return builder

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
                except ValueError as exc:  # pragma: no cover - only triggered on malformed files
                    raise RuntimeError(f"Invalid time value '{time_elem.attrib['Value']}' in grid '{grid_name}'.") from exc
                step_id = f"time:{time_value}"
            elif step_id is None:
                step_id = current_path
            grid_type = (grid.attrib.get('GridType', 'Uniform') or '').strip().lower()
            if grid_type == 'collection':
                child_context = TraverseContext(time_value=time_value, step_id=step_id)
                for child in grid.findall('Grid'):
                    walk(child, child_context, current_path_items)
            else:
                builder = get_builder(step_id, time_value)
                uniform = self._parse_uniform_grid(grid, current_path)
                if len(uniform.node_dims) != self.mesh.ndim:
                    raise RuntimeError(f"Grid '{current_path}' dimensionality mismatch.")
                offset = self._offset_resolver.resolve(uniform.name, uniform.node_dims, uniform.origin)
                builder.add_grid(uniform, offset)

        for top_grid in self.domain.findall('Grid'):
            walk(top_grid, TraverseContext(time_value=None, step_id=None), [])
        return [builders[key].build() for key in order]

    @property
    def has_explicit_times(self) -> bool:
        return all(snapshot.time_value is not None for snapshot in self.snapshots)


def _ensure_mesh_compatibility(mesh_a: MeshInfo, mesh_b: MeshInfo) -> None:
    if mesh_a.topology_type != mesh_b.topology_type:
        raise RuntimeError("Topology types differ between the two files.")
    if mesh_a.node_shape != mesh_b.node_shape:
        raise RuntimeError("Topology dimensions differ between the two files.")
    if not np.allclose(mesh_a.origin, mesh_b.origin):
        raise RuntimeError("Origin vectors differ between the two files.")
    if not np.allclose(mesh_a.spacing, mesh_b.spacing):
        raise RuntimeError("Grid spacing differs between the two files.")


def _align_snapshots(series_a: XdmfSeries, series_b: XdmfSeries) -> List[Tuple[Snapshot, Snapshot]]:
    if len(series_a.snapshots) != len(series_b.snapshots):
        raise RuntimeError("The two files contain a different number of time steps.")
    if series_a.has_explicit_times and series_b.has_explicit_times:
        pairs: List[Tuple[Snapshot, Snapshot]] = []
        used: set[int] = set()
        for snap_a in series_a.snapshots:
            match_idx = None
            for idx, snap_b in enumerate(series_b.snapshots):
                if idx in used:
                    continue
                if math.isclose(snap_a.time_value, snap_b.time_value, rel_tol=1e-9, abs_tol=1e-12):
                    match_idx = idx
                    break
            if match_idx is None:
                raise RuntimeError(f"No matching time value found for step at {snap_a.time_value}.")
            used.add(match_idx)
            pairs.append((snap_a, series_b.snapshots[match_idx]))
        return pairs
    if series_a.has_explicit_times != series_b.has_explicit_times:
        raise RuntimeError("Only one file defines explicit times; cannot align snapshots.")
    return list(zip(series_a.snapshots, series_b.snapshots))


def _compare_fields(time_label: str, center_label: str, fields_a: Dict[str, np.ndarray], fields_b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
        l2 = float(np.linalg.norm(diff.ravel(), ord=2))
        linf = float(np.max(np.abs(diff))) if diff.size else 0.0
        print(f"{time_label}: {center_label} field '{name}': L2={l2:.6e}, Linf={linf:.6e}")
        diffs[name] = diff
    return diffs


def _write_diff_series(output_path: str, mesh: MeshInfo, snapshots: List[Snapshot]) -> None:
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


def compare_series(path_a: str, path_b: str, diff_out: Optional[str]) -> None:
    series_a = XdmfSeries(path_a)
    series_b = XdmfSeries(path_b)
    _ensure_mesh_compatibility(series_a.mesh, series_b.mesh)
    print('Mesh topology matches - safe to compare fields.')
    pairs = _align_snapshots(series_a, series_b)
    diff_snapshots: List[Snapshot] = []
    for snap_a, snap_b in pairs:
        label = snap_a.label
        diff_snapshot = Snapshot(step_id=snap_a.step_id, index=snap_a.index, time_value=snap_a.time_value)
        node_diff = _compare_fields(label, 'point', snap_a.node_fields, snap_b.node_fields)
        cell_diff = _compare_fields(label, 'cell', snap_a.cell_fields, snap_b.cell_fields)
        diff_snapshot.node_fields = node_diff
        diff_snapshot.cell_fields = cell_diff
        if node_diff or cell_diff:
            diff_snapshots.append(diff_snapshot)
    if diff_out and diff_snapshots:
        _write_diff_series(diff_out, series_a.mesh, diff_snapshots)
    elif diff_out:
        print('No overlapping fields were found; skipping diff file export.')


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare fields stored in XDMF files.')
    parser.add_argument('file_a', help='Reference XDMF file (e.g., serial output).')
    parser.add_argument('file_b', help='XDMF file to compare (e.g., parallel output).')
    parser.add_argument('diff_out', nargs='?', help='Optional XDMF path for writing field differences.')
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        compare_series(args.file_a, args.file_b, args.diff_out)
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
