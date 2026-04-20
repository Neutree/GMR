import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def _apply_bvh_hierarchy_offset(local_pos, bones, hierarchy_offset):
    """Apply per-joint local offsets in BVH local coordinates before FK.

    Args:
        local_pos: ndarray [F, J, 3], BVH local positions in centimeters.
        bones: list of joint names.
        hierarchy_offset: dict of joint -> [x, y, z] in meters.
    """
    if not hierarchy_offset:
        return local_pos

    pos = local_pos.copy()
    bone_to_idx = {name: i for i, name in enumerate(bones)}
    for bone_name, offset_m in hierarchy_offset.items():
        idx = bone_to_idx.get(bone_name)
        if idx is None:
            continue
        # BVH read data uses centimeters; config offsets use meters.
        pos[:, idx, :] = pos[:, idx, :] + np.asarray(offset_m, dtype=np.float32) * 100.0
    return pos


def load_bvh_file(bvh_file, format="lafan1", hierarchy_offset=None, return_original_frames=False, up_axis="y"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    local_pos = _apply_bvh_hierarchy_offset(data.pos, data.bones, hierarchy_offset)
    global_data = utils.quat_fk(data.quats, local_pos, data.parents)

    if up_axis == "y":
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    else:
        rotation_matrix = np.eye(3)
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    original_frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        original_result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]

            # Original BVH overlay frame in the same world-up convention (z-up)
            # so it can be directly compared with converted/retargeted data.
            original_position = global_data[1][frame, i] @ rotation_matrix.T / 100
            original_orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            original_result[bone] = [original_position, original_orientation]
            
        if format == "lafan1":
            # Add modified foot pose
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        elif format == "noitom":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
        original_frames.append(original_result)
    
    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    if return_original_frames:
        return frames, human_height, original_frames
    return frames, human_height


