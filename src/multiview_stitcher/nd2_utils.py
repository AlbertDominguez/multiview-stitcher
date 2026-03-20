from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import nd2
except ImportError:
    nd2 = None


def _check_nd2():
    if nd2 is None:
        raise ImportError(
            "nd2 is required to read ND2 files. "
            "Please install it using `pip install multiview-stitcher[nd2]` or `pip install nd2`."
        )


def get_nd2_shape(filepath):
    """
    Get the shape of an ND2 file.
    """
    _check_nd2()

    filepath = Path(filepath)
    with nd2.ND2File(str(filepath)) as nd2_file:
        return OrderedDict(nd2_file.sizes)


def get_spacing_from_nd2(filepath):
    """
    Get pixel spacing (micrometers) from an ND2 file.
    """
    _check_nd2()

    filepath = Path(filepath)
    with nd2.ND2File(str(filepath)) as nd2_file:
        voxel = nd2_file.voxel_size()
        spacing = {
            "x": float(voxel.x) if voxel.x else 1.0,
            "y": float(voxel.y) if voxel.y else 1.0,
        }

        if "Z" in nd2_file.sizes:
            spacing["z"] = float(voxel.z) if voxel.z else 1.0

    return spacing


def get_nd2_channel_names(filepath):
    """
    Get channel names of an ND2 file.
    """
    _check_nd2()

    filepath = Path(filepath)
    with nd2.ND2File(str(filepath)) as nd2_file:
        xim = nd2_file.to_xarray(delayed=True, squeeze=False)
        if "C" not in xim.coords:
            return []
        return list(xim.coords["C"].values)


def get_nd2_mosaic_origins(filepath, spacing):
    """
    Return per-tile origins in micrometers for an ND2 file.

    For ND2 we use XYPosLoop stage positions when available.
    """
    _check_nd2()

    filepath = Path(filepath)
    with nd2.ND2File(str(filepath)) as nd2_file:
        n_pos = int(nd2_file.sizes.get("P", 1))

        has_z = "z" in spacing
        default_origin = {"y": 0.0, "x": 0.0}
        if has_z:
            default_origin["z"] = 0.0

        xy_loop = next(
            (
                loop
                for loop in nd2_file.experiment
                if getattr(loop, "type", None) == "XYPosLoop"
            ),
            None,
        )

        if xy_loop is None:
            return [default_origin.copy() for _ in range(n_pos)]

        points = getattr(xy_loop.parameters, "points", [])
        if not points:
            return [default_origin.copy() for _ in range(n_pos)]

        origins = []
        for i_pos in range(n_pos):
            if i_pos < len(points):
                point = points[i_pos]
                origin = {
                    "y": float(point.stagePositionUm.y),
                    "x": float(point.stagePositionUm.x),
                }
                if has_z:
                    origin["z"] = float(point.stagePositionUm.z)
            else:
                origin = default_origin.copy()
            origins.append(origin)

    return origins


def read_nd2_into_xims(filepath, scene_index=0):
    """
    Read ND2 data into a list of xarray DataArrays, one per mosaic tile/position.

    ND2 does not expose a scene concept comparable to CZI/LIF, so only
    scene_index=0 is currently supported.
    """
    _check_nd2()

    if scene_index != 0:
        raise ValueError("ND2 reader currently supports only scene_index=0")

    filepath = Path(filepath)

    xims = []
    with nd2.ND2File(str(filepath)) as nd2_file:
        n_pos = int(nd2_file.sizes.get("P", 1))

        voxel = nd2_file.voxel_size()
        spacing = {
            "x": float(voxel.x) if voxel.x else 1.0,
            "y": float(voxel.y) if voxel.y else 1.0,
        }
        if "Z" in nd2_file.sizes:
            spacing["z"] = float(voxel.z) if voxel.z else 1.0

        for p in range(n_pos):
            xim = nd2_file.to_xarray(delayed=True, squeeze=False, position=p)

            # Keep dimensions compatible with multiview-stitcher conventions.
            drop_dims = [
                dim
                for dim in xim.dims
                if dim not in ["T", "C", "Z", "Y", "X"]
            ]
            if drop_dims:
                xim = xim.squeeze(drop_dims, drop=True)

            rename_map = {
                dim: dim.lower()
                for dim in xim.dims
                if dim in ["T", "C", "Z", "Y", "X"]
            }
            xim = xim.rename(rename_map)

            target_dims = [d for d in ["t", "c", "z", "y", "x"] if d in xim.dims]
            xim = xim.transpose(*target_dims)

            spatial_dims = [d for d in ["z", "y", "x"] if d in xim.dims]
            spatial_coords = {
                d: np.arange(xim.sizes[d], dtype=float) * spacing[d]
                for d in spatial_dims
            }
            xim = xim.assign_coords(spatial_coords)

            xims.append(xim)

    return xims
