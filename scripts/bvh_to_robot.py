import argparse
import pathlib
import time
import json
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import VIEWER_CAM_DISTANCE_DICT, IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
import mujoco as mj

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "noitom"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "pnd_adam_pro", "agibot_x2"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--pause_frame",
        action="store_true",
        default=False,
        help="Pause after each frame for debugging.",
    )

    parser.add_argument(
        "--follow_camera",
        action="store_true",
        default=False,
        help="Whether to make the camera follow the robot root.",
    )

    parser.add_argument(
        "--show_original_human_frame",
        action="store_true",
        default=False,
        help="Overlay original BVH human frames (larger and transparent) for coordinate debugging.",
    )

    parser.add_argument(
        "--original_human_axis_scale",
        default=0.16,
        type=float,
        help="Axis size for original BVH frame overlay.",
    )

    parser.add_argument(
        "--original_human_alpha",
        default=0.15,
        type=float,
        help="Transparency for original BVH frame overlay.",
    )

    parser.add_argument(
        "--up-axis",
        choices=["y", "z"],
        default="y",
        help="The up axis of the input BVH file. The viewer uses z-up convention, so if the input BVH is y-up, it will be converted to z-up for correct visualization and retargeting.",
    )

    parser.add_argument(
        "--show_body_frame",
        action="store_true",
        default=False,
        help="Show body frame in the viewer (instead of inertial/world frame).",
    )
    parser.add_argument(
        "--ground_height",
        default = None,
        type=float,
        help="override ground_height config in json file"
    )

    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    src_human = f"bvh_{args.format}"
    ik_config_path = IK_CONFIG_DICT[src_human][args.robot]
    with open(ik_config_path) as f:
        ik_config = json.load(f)
    if args.ground_height is not None:
        ik_config["ground_height"] = args.ground_height
    bvh_hierarchy_offset = ik_config.get("bvh_hierarchy_offset", {})

    # Load BVH trajectory with optional pre-FK hierarchy offsets.
    if args.show_original_human_frame:
        lafan1_data_frames, actual_human_height, original_human_data_frames = load_bvh_file(
            args.bvh_file,
            format=args.format,
            hierarchy_offset=bvh_hierarchy_offset,
            return_original_frames=True,
            up_axis=args.up_axis,
        )
    else:
        lafan1_data_frames, actual_human_height = load_bvh_file(
            args.bvh_file,
            format=args.format,
            hierarchy_offset=bvh_hierarchy_offset,
            return_original_frames=False,
            up_axis=args.up_axis,
        )
        original_human_data_frames = None
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=src_human,
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )
    retargeter.ground_offset = ik_config.get("ground_height", 0.0)

    motion_fps = args.motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    if args.show_body_frame:
        import mujoco as mj
        robot_motion_viewer.viewer.opt.frame = mj.mjtFrame.mjFRAME_BODY

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    
    if not args.follow_camera:
        robot_motion_viewer.viewer.cam.lookat = [0, 0, 0.5]
        robot_motion_viewer.viewer.cam.distance = VIEWER_CAM_DISTANCE_DICT[args.robot]
        robot_motion_viewer.viewer.cam.elevation = -10  # 正面视角，轻微向下看

    while robot_motion_viewer.viewer.is_running():
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)

        original_human_overlay = None
        if original_human_data_frames is not None:
            original_frame = original_human_data_frames[i]
            root_name = retargeter.human_root_name
            if root_name in original_frame and root_name in retargeter.scaled_human_data:
                root_shift = (
                    retargeter.scaled_human_data[root_name][0]
                    - original_frame[root_name][0]
                )
            else:
                root_shift = np.zeros(3)

            original_human_overlay = {
                body_name: [pos + root_shift, rot]
                for body_name, (pos, rot) in original_frame.items()
            }
        

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            human_motion_data_original=original_human_overlay,
            original_human_point_scale=args.original_human_axis_scale,
            original_human_alpha=args.original_human_alpha,
            rate_limit=args.rate_limit,
            follow_camera=args.follow_camera,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
        if args.pause_frame:
            input("press to view next frame")
        if args.save_path is not None:
            qpos_list.append(qpos)
    
    if args.save_path is not None:
        import pickle
        
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        # Get joint names from mujoco model
        model = robot_motion_viewer.model
        joint_names = []
        for i in range(model.njnt):
            # skip root joint (free joint)
            if i == 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
                continue
            joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            joint_names.append(joint_name)
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
            "joint_names": joint_names
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
