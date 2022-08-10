import numpy as np

#末端姿态偏差的定义
def orientation_error(desired, current):#姿态矩阵的偏差3×3的
    rc1 = current[0:3, 0]#提取前三行作为第一列
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]
    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))#cross是叉乘
    return error

#robot:机器人计算的KDL的类  sim:mujoco初始化的仿真环境  k: 刚度 d:阻尼  desired_pos:期望的位置  desired_ori:期望的姿态  tau_last：传入一个力矩 eef_nam：对应机器人末端位置的site名称
def torque_cartesian(robot, sim, k, d, eef_name, desired_pos, desired_ori):
    M = robot.mass_matrix()
    qd = np.array(sim.data.qvel[:])
    J = robot.jacobian()
    J_inv = np.linalg.inv(J)#雅各比矩阵的逆
    Jd = robot.jacobian_dot()#雅各比矩阵的微分
    Md = np.dot(J_inv.T, np.dot(M, J_inv))#目标质量矩阵，在讲解里边
    tau = sim.data.qfrc_bias[:]
    #获取末端的位置/姿态/速度/
    x_pos = np.array(sim.data.get_site_xpos(eef_name))
    x_ori = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]))
    x_pos_vel = np.array(sim.data.site_xvelp[sim.model.site_name2id(eef_name)])
    x_ori_vel = np.array(sim.data.site_xvelr[sim.model.site_name2id(eef_name)])

    coef = np.dot(M, J_inv)
    xd_error = np.concatenate([-x_pos_vel, -x_ori_vel])#末端姿态和位置的拼接
    sum = np.multiply(d, xd_error)
    pos_error = desired_pos - x_pos#位置偏差
    ori_error = orientation_error(desired_ori, x_ori)#姿态偏差
    x_error = np.concatenate([pos_error, ori_error])#两者拼接

    sum += np.multiply(k, x_error)
    sum -= np.dot(np.dot(Md, Jd), qd)
    tau += np.dot(coef, sum)

    return tau
