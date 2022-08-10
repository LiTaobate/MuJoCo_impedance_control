import utils
import numpy as np
import ikfastpy


solver = ikfastpy.PyKinematics()
n_joints = solver.getDOF()


def ik(rbt, init, pos, ori):
    pos, ori, ok = utils.check_pos_ori_valid(pos, ori)
    if not ok:
        return []
    ee_pos = np.append(ori, pos, axis=1)
    joint_configs = solver.inverse(ee_pos.reshape(-1).tolist())
    n_solutions = int(len(joint_configs) / n_joints)
    joint_configs = np.asarray(joint_configs).reshape(n_solutions, n_joints)
    min_norm = np.inf
    res = []
    for i in range(n_solutions):
        norm = np.linalg.norm(joint_configs[i] - init)
        if norm < min_norm:
            res = joint_configs[i]
            min_norm = norm
    if n_solutions == 0:
        res = rbt.get_ik(init, pos, ori)
        if sum(abs(res - init)) > 3 / 180 * np.pi:
            res = []
    return res

#robot:机器人计算的KDL的类  sim:mujoco初始化的仿真环境  k: 刚度 d:阻尼  desired_pos:期望的位置  desired_ori:期望的姿态  tau_last：传入一个力矩 
def torque_joint(robot, sim, k, d, desired_pos, desired_ori, tau_last):
    q = np.array(sim.data.qpos[:])#当前关节的位置
    qd = np.array(sim.data.qvel[:])#速度
    q_target = ik(robot, q, desired_pos, desired_ori)#计算当前关节的目标位置
    M = robot.mass_matrix()#机器人的质量矩阵
    ok = False
    tau = tau_last
    if len(q_target) > 0:
        #robot的关节空间控制的计算公式（multiply等同于向量相乘）
        tau = np.multiply(k, q_target - q) - np.multiply(d, qd)
        tau = np.dot(M, tau)#乘上质量矩阵会更稳定一些
        tau += robot.coriolis_gravity()#加上科氏力和重力矩
        ok = True #标记用于判断解算是否成功
    return tau, ok