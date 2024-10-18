import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    initial_positions = meta_data.joint_initial_position
    joint_parent = meta_data.joint_parent
    joint_name = meta_data.joint_name
    origin_offset = np.array([initial_positions[i] - initial_positions[joint_parent[i]] if joint_parent[i] != -1 else [0,0,0] for i in range(len(joint_parent))])
    path_offset = np.array([initial_positions[path[i]] - initial_positions[path[i-1]] if i>0 else [0,0,0] for i in range(len(path))])
    origin_eulers = np.array([(R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_euler('XYZ') if joint_parent[i] != -1 else [0,0,0] for i in range(len(joint_parent))])
    path_eulers = np.array([(R.from_quat(joint_orientations[path[i]]).inv() * R.from_quat(joint_orientations[path[i-1]])).as_euler('XYZ') if i>0 else [0,0,0] for i in range(len(path))])
    for i in range(100):
        for j in range(len(path) - 1):
            target_dir = (target_pose - joint_positions[path[j]])
            target_dir = target_dir / np.linalg.norm(target_dir)
            end_dir = (joint_positions[path[-1]] - joint_positions[path[j]])
            end_dir = end_dir / np.linalg.norm(end_dir)
            rotvec = -np.cross(target_dir, end_dir) * 0.025
            path_eulers[j] = (R.from_rotvec(rotvec) * R.from_euler('XYZ', path_eulers[j])).as_euler('XYZ')
        for j in range(len(path)):
            if j==0:
                joint_orientations[path[j]] = (R.from_euler('XYZ', path_eulers[j])).as_quat()
                continue
            joint_orientations[path[j]] = (R.from_quat(joint_orientations[path[j-1]]) * R.from_euler('XYZ', path_eulers[j])).as_quat()
            joint_positions[path[j]] = joint_positions[path[j-1]] + R.from_quat(joint_orientations[path[j-1]]).as_matrix() @ path_offset[j]

        if np.linalg.norm(joint_positions[path[-1]] - target_pose) < 0.01:
            break

    #print(np.linalg.norm(joint_positions[path[-1]] - target_pose))

    #print(len(joint_positions), len(joint_parent), origin_eulers.shape)
    index = 0
    for j in range(len(joint_parent)):
        if joint_parent[j] != -1 and not j in path:
            joint_orientations[j] = (R.from_quat(joint_orientations[joint_parent[j]]) * R.from_euler('XYZ', origin_eulers[j])).as_quat()
            joint_positions[j] = joint_positions[joint_parent[j]] + R.from_quat(joint_orientations[joint_parent[j]]).as_matrix() @ origin_offset[j]
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pose = joint_positions[meta_data.joint_name.index('RootJoint')] + np.array([relative_x, 0, relative_z])
    target_pose[1] = target_height

    return part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations