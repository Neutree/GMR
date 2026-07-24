import numpy as np
from scipy.spatial.transform import Rotation as R
import re

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def get_bvh_frame_info(bvh_file):
    """Read BVH header fps metadata without loading the full motion.

    Returns:
        dict with keys:
          - frames: int | None
          - frame_time: float (seconds)
          - fps: float (= 1 / frame_time)
    """
    frames = None
    frame_time = None
    with open(bvh_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_frames = re.match(r"\s*Frames:\s+(\d+)", line)
            if m_frames:
                frames = int(m_frames.group(1))
                continue
            m_time = re.match(r"\s*Frame Time:\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", line)
            if m_time:
                frame_time = float(m_time.group(1))
                break

    if frame_time is None or frame_time <= 0:
        raise ValueError(f"Cannot read valid 'Frame Time' from BVH: {bvh_file}")

    fps = 1.0 / frame_time
    return {
        "frames": frames,
        "frame_time": frame_time,
        "fps": fps,
    }


def resolve_motion_fps(bvh_file, motion_fps=None):
    """Resolve motion fps from BVH header unless an explicit override is given.

    Args:
        bvh_file: path to BVH
        motion_fps: optional explicit fps override (int/float). None/<=0 means auto.

    Returns:
        (fps_int, info_dict)
    """
    info = get_bvh_frame_info(bvh_file)
    auto_fps = int(round(info["fps"]))
    if motion_fps is None or int(motion_fps) <= 0:
        used = auto_fps
        source = "auto(from BVH Frame Time)"
    else:
        used = int(motion_fps)
        source = "manual override"
    print(
        f"[BVH FPS] file={bvh_file}\n"
        f"         Frames: {info['frames']}\n"
        f"         Frame Time: {info['frame_time']:.6f} s\n"
        f"         computed fps = 1/FrameTime = {info['fps']:.6f}\n"
        f"         >>> using motion_fps = {used}  ({source})"
    )
    return used, info


def _infer_initial_forward(frame, root_name):
    up_axis = np.array([0.0, 0.0, 1.0])
    lateral_pairs = [
        ("LeftShoulder", "RightShoulder"),
        ("LeftArm", "RightArm"),
        ("LeftUpLeg", "RightUpLeg"),
        ("LeftLeg", "RightLeg"),
    ]

    for left_name, right_name in lateral_pairs:
        if left_name not in frame or right_name not in frame:
            continue
        lateral = np.asarray(frame[left_name][0], dtype=np.float64) - np.asarray(frame[right_name][0], dtype=np.float64)
        lateral[2] = 0.0
        lateral_norm = np.linalg.norm(lateral[:2])
        if lateral_norm < 1e-8:
            continue
        lateral /= lateral_norm
        forward = np.cross(lateral, up_axis)
        forward_norm = np.linalg.norm(forward[:2])
        if forward_norm >= 1e-8:
            return forward / forward_norm

    root_quat = frame[root_name][1]
    root_rot = R.from_quat(root_quat, scalar_first=True)
    forward = root_rot.apply(np.array([1.0, 0.0, 0.0]))
    forward[2] = 0.0
    forward_norm = np.linalg.norm(forward[:2])
    if forward_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return forward / forward_norm


def _compute_world_up_alignment(frame, root_name):
    forward = _infer_initial_forward(frame, root_name)
    yaw = np.arctan2(forward[1], forward[0])
    return R.from_euler("z", -yaw)


def _normalize_frames_to_origin(frames, root_name):
    if not frames or root_name not in frames[0]:
        return frames

    root_pos, _ = frames[0][root_name]
    translation = np.array([root_pos[0], root_pos[1], 0.0], dtype=np.float64)
    alignment = _compute_world_up_alignment(frames[0], root_name)

    normalized_frames = []
    for frame in frames:
        normalized_frame = {}
        for body_name, (position, orientation) in frame.items():
            shifted_position = np.asarray(position, dtype=np.float64) - translation
            aligned_position = alignment.apply(shifted_position)
            aligned_orientation = (
                alignment * R.from_quat(orientation, scalar_first=True)
            ).as_quat(scalar_first=True)
            normalized_frame[body_name] = [aligned_position, aligned_orientation]
        normalized_frames.append(normalized_frame)

    return normalized_frames


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


def load_bvh_file(
    bvh_file,
    format="lafan1",
    hierarchy_offset=None,
    return_original_frames=False,
    up_axis="y",
    normalize_to_origin=False,
    root_name=None,
):
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
    global_data_origin = utils.quat_fk(data.quats, data.pos, data.parents)

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
            original_position = global_data_origin[1][frame, i] @ rotation_matrix.T / 100
            original_orientation = utils.quat_mul(rotation_quat, global_data_origin[0][frame, i])
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

    if normalize_to_origin:
        target_root_name = root_name or data.bones[0]
        frames = _normalize_frames_to_origin(frames, target_root_name)
        if return_original_frames:
            original_frames = _normalize_frames_to_origin(original_frames, target_root_name)

    if return_original_frames:
        return frames, human_height, original_frames
    return frames, human_height


