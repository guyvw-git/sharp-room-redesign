"""PLY I/O utilities for 3DGS Gaussian splat files.

Handles the MetalSplat2 compact format:
  element vertex N  — 14 float properties (x,y,z,f_dc_0/1/2,opacity,scale_0/1/2,rot_0/1/2/3)
  element extrinsic 16
  element intrinsic 9
  element image_size 2
  element frame 2
  element disparity 2
  element color_space 1
  element version 3
"""
import struct
import numpy as np
import os


TYPE_SIZES = {
    'float': 4, 'double': 8,
    'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4,
    'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
    'char': 1, 'uchar': 1, 'int8': 1, 'uint8': 1,
}


def parse_header(path):
    """Parse PLY header. Returns full info about all elements."""
    with open(path, 'rb') as f:
        header_bytes = b''
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b'end_header':
                break
        data_offset = f.tell()

    header_str = header_bytes.decode('ascii', errors='ignore')
    lines = header_str.strip().split('\n')

    elements = []  # list of (name, count, [(type, propname), ...])
    current_elem = None

    for line in lines:
        line = line.strip()
        if line.startswith('element '):
            parts = line.split()
            current_elem = {'name': parts[1], 'count': int(parts[2]), 'properties': []}
            elements.append(current_elem)
        elif line.startswith('property ') and current_elem is not None:
            parts = line.split()
            current_elem['properties'].append((parts[1], parts[2]))

    # Find vertex element
    vertex_elem = next((e for e in elements if e['name'] == 'vertex'), None)
    if vertex_elem is None:
        raise ValueError("No vertex element found in PLY")

    vertex_count = vertex_elem['count']
    properties = [name for (ptype, name) in vertex_elem['properties']
                  if ptype in TYPE_SIZES]

    return header_bytes, vertex_count, properties, data_offset, elements


def read_ply(path):
    """Read a 3DGS PLY file.
    Returns (vertex_data, header_bytes, properties, tail_bytes).
    tail_bytes = all bytes after vertex data (other elements).
    """
    header_bytes, n_verts, props, data_offset, elements = parse_header(path)

    # Vertex element: all floats
    n_props = len(props)
    vertex_bytes = n_verts * n_props * 4  # all float32

    with open(path, 'rb') as f:
        f.seek(data_offset)
        raw = f.read(vertex_bytes)
        tail_bytes = f.read()  # remaining elements

    data = np.frombuffer(raw, dtype=np.float32).reshape(n_verts, n_props).copy()
    return data, header_bytes, props, tail_bytes


def write_ply(path, data, header_bytes, tail_bytes=b''):
    """Write a 3DGS PLY file, preserving header and tail elements."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Update vertex count in header if data changed
    header_str = header_bytes.decode('ascii', errors='ignore')
    n_verts = len(data)
    lines = header_str.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('element vertex'):
            new_lines.append(f'element vertex {n_verts}')
        else:
            new_lines.append(line)
    # Rebuild header bytes — keep same byte length if possible
    new_header_str = '\n'.join(new_lines)
    if not new_header_str.endswith('\n'):
        new_header_str += '\n'
    new_header_bytes = new_header_str.encode('ascii')

    with open(path, 'wb') as f:
        f.write(new_header_bytes)
        f.write(data.astype(np.float32).tobytes())
        f.write(tail_bytes)


def read_positions(path):
    """Fast read of just x,y,z positions from PLY."""
    header_bytes, n_verts, props, data_offset, elements = parse_header(path)
    n_props = len(props)
    vertex_bytes = n_verts * n_props * 4

    xi = props.index('x') if 'x' in props else 0
    yi = props.index('y') if 'y' in props else 1
    zi = props.index('z') if 'z' in props else 2

    with open(path, 'rb') as f:
        f.seek(data_offset)
        raw = f.read(vertex_bytes)

    data = np.frombuffer(raw, dtype=np.float32).reshape(n_verts, n_props)
    return data[:, [xi, yi, zi]], props, n_verts
