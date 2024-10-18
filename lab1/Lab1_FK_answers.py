from math import degrees
from re import A, T
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    with open(bvh_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    index = -1
    stack = [-1]
    joint_name = []
    joint_parent = []
    joint_offset = []

    for line in lines:
        line = line.strip()
        tokens = line.split()
        if tokens[0] == "ROOT" or tokens[0] == "JOINT":
            joint_parent.append(stack[-1])
            joint_name.append(tokens[1])
        elif tokens[0] == "End":
            joint_parent.append(stack[-1])
            joint_name.append(joint_name[stack[-1]] + "_end")
        elif tokens[0] == "{":
            index += 1
            stack.append(index)
        elif tokens[0] == "}" and len(stack) > 0:
            stack.pop()
        elif "OFFSET" in line:
            offset = [float(tokens[i]) for i in range(1, 4)]
            joint_offset.append(offset)
        elif "MOTION" in line:
            break

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """   
    joint_positions = []
    joint_orientations = []


    index = 1
    for i in range(len(joint_name)):

        if joint_name[i].endswith("_end"):
            joint_positions.append(parent_position + R.from_quat(parent_orientation).as_matrix() @ joint_offset[i])
            joint_orientations.append(joint_orientations[joint_parent[i]])
            continue

        gamma, beta, alpha = np.deg2rad(motion_data[frame_id][3*index:3*index+3])
                
        R_x = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])

        R_y = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        R_z = np.array([
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)]
        ])

        if joint_parent[i] == -1:
            #print(i, index, joint_name[i])
            joint_positions.append(joint_offset[i] + motion_data[frame_id][:3])
            joint_orientations.append(R.from_matrix(R_z @ R_y @ R_x).as_quat())
            index += 1
        elif not joint_name[i].endswith("_end"):
            parent_position = joint_positions[joint_parent[i]]
            parent_orientation = joint_orientations[joint_parent[i]]
            joint_positions.append(parent_position + R.from_quat(parent_orientation).as_matrix() @ joint_offset[i])
            joint_orientations.append(R.from_matrix(R.from_quat(parent_orientation).as_matrix()
                                                    @ R_z @ R_y @ R_x).as_quat())
            index += 1

    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_name, T_parent, T_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_name, A_parent, A_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion = load_motion_data(A_pose_bvh_path)
    motion_data = np.zeros((A_motion.shape[0], len(T_name) * 3 + 3), dtype=np.float64)
    motion_source = np.zeros(len(T_name)*3+3, dtype=np.int32)

    jindex = 0
    ls_index = rs_index = -1
    for j in range(len(A_name)):
        if j == 0:
            motion_source[0:6] = [0, 1, 2, 3, 4, 5]
            jindex += 1
        else:
            if A_name[j] in T_name and not A_name[j].endswith("_end"):
                index = -1
                for k in range(len(T_name)):
                    if not T_name[k].endswith("_end"):
                        index += 1
                    if T_name[k] == A_name[j]:
                        break
                if not index == -1:
                    motion_source[index * 3 + 3: index * 3 + 6] = [jindex * 3 + 3, jindex * 3 + 4, jindex * 3 + 5]
                if A_name[j] == 'lShoulder':
                    ls_index = index
                if A_name[j] == 'rShoulder':
                    rs_index = index
                jindex += 1
    
    for i in range(A_motion.shape[0]):
        motion_data[i][0:3] = A_motion[i][0:3]
        for j in range(len(T_name)):
            motion_data[i][j * 3 + 3: j * 3 + 6] = A_motion[i][motion_source[j * 3 + 3: j * 3 + 6]]
            if j == ls_index:
                motion_data[i][j*3+3:j*3+6] = (R.from_euler('XYZ', [0, 0, -45], degrees=True) 
                                               * R.from_euler('XYZ', motion_data[i][j*3+3:j*3+6], degrees=True)).as_euler('XYZ', degrees=True)
                #motion_data[i][j*3+5] -= 45
            if j == rs_index:
                motion_data[i][j*3+3:j*3+6] = (R.from_euler('XYZ', [0, 0, +45], degrees=True) 
                                               * R.from_euler('XYZ', motion_data[i][j*3+3:j*3+6], degrees=True)).as_euler('XYZ', degrees=True)
                #motion_data[i][j*3+5] += 45

    return motion_data
