import mujoco_py as mp
import func_mj 
# from joint_ipd import torque_joint
from cartesian_ipd import torque_cartesian
import func
import numpy as np
import math
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore


#定义一些存储末端位置的量（300长度的list）
end_x = [0] * 300
end_y = [0] * 300
end_z = [0] * 300

#更新数据：把最旧的数据舍弃掉，然后
def update_x():
    global end_x, curve_x
    end_x[:-1] = end_x[1:]
    end_x[-1] = sim.data.get_site_xpos('ee')[0]
    curve_x.setData(end_x)


def update_y():
    global end_y, curve_y
    end_y[:-1] = end_y[1:]
    end_y[-1] = sim.data.get_site_xpos('ee')[1]
    curve_y.setData(end_y)

def update_z():
    global end_z, curve_z
    end_z[:-1] = end_z[1:]
    end_z[-1] = sim.data.get_site_xpos('ee')[2]
    curve_z.setData(end_z)


def update_xy():
    curve_xy.setData(end_x, end_y)


if __name__ == '__main__':
    model = mp.load_model_from_path('ur5.xml')#读取mujoco模型
    sim = mp.MjSim(model)#初始化仿真环境
    viewer = mp.MjViewer(sim)#渲染显示
    # Rbt = func.Robot(sim, 'ee')#初始化一个包括kdl和mujoco的计算的类

    Rbt = func.Robot(sim, 'ee')
    init = [0, -1.6, 1.6, -1.6, -1.6, 0]#初始位姿和关节角度
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    for i in range(6):
        sim.data.set_joint_qpos(joint_names[i], init[i])#一一对应关节
    sim.forward()
    pos, ori = Rbt.fk()#获得末端姿态 c
    desired_pos = pos.copy()#期望的位置
    desired_ori = ori.copy()#期望的姿态 
    kj, dj = np.array([20] * 6, dtype=np.float32), np.array([100] * 6, dtype=np.float32)#关节空间的参数
    kc = np.array([300, 300, 300, 600, 600, 600], dtype=np.float32)# 迪卡尔空间
    dc = np.array([120, 120, 120, 70, 70, 70], dtype=np.float32)
    last_tau = 0#关节空间前一次的值
    steps = 0
    body_id = sim.model.body_name2id('wrist_3_link')#读到最后一个连杆的ID

    #显示部分的主要程序
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="End Effector Position")
    win.resize(800, 200)#窗口大小
    win.setWindowTitle('End Effector Position')#窗口标题
    pg.setConfigOptions(antialias=True)

    timer = QtCore.QTimer()#开启定时器c
    pz = win.addPlot(title="Position-Z")
    curve_z = pz.plot(pen='y')
    circle = win.addPlot(title='Position-XY')
    curve_xy = circle.plot(pen='r')

    timer.timeout.connect(update_x)
    timer.timeout.connect(update_y)
    timer.timeout.connect(update_z) 
    timer.timeout.connect(update_xy)
    timer.start(1)#1毫秒更新一次
    

    while 1:
        # #下边两行对应轨迹的写法（0和1分别代表X 维度的值）
        desired_pos[0] = pos[0] + 0.1 * math.cos(steps / 180 * np.pi)#代表一步走圆的一度
        desired_pos[1] = pos[1] + 0.1 * math.sin(steps / 180 * np.pi)
        # tau, ok = torque_joint(Rbt, sim, kj, dj, desired_pos, desired_ori, last_tau)
        tau = torque_cartesian(Rbt, sim, kc, dc, 'ee', desired_pos, desired_ori) #关节的驱动力矩

        #以下是给机器人施加外力的
        if 3000 <= steps < 3100:
            sim.data.xfrc_applied[body_id][0] = 100   #表示连杆所受外力的力矩，第一维是每个body的ID，第二维有六个量（x，y，z方向的受力，后边是力矩）
        else:
            sim.data.xfrc_applied[body_id][0] = 0     #还原数值，不施加外力
        sim.data.ctrl[:] = tau    #输入力矩，输入到控制器上
        last_tau = tau.copy()
        sim.step()
        steps += 1
        viewer.render()
