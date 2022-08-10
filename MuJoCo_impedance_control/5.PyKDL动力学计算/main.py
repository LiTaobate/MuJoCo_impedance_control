import mujoco_py as mp
import func_mj#导入的是mujoco的运动学包
import func_kdl#导入的是kdl的运动学包

if __name__ == '__main__':
    model = mp.load_model_from_path('ur5.xml')
    sim = mp.MjSim(model)
    viewer = mp.MjViewer(sim)

    RbtMj = func_mj.RobotMj(sim, 'ee')
    # RbtKdl = func_kdl.RobotKdl(sim)
    #初始关节位置，关节弧度值
    initial_qpos = {
        'shoulder_pan_joint': -3.1,
        'shoulder_lift_joint': -1.6,
        'elbow_joint': 1.6,
        'wrist_1_joint': -1.6,
        'wrist_2_joint': -1.6,
        'wrist_3_joint': 0,
    }
    #将初始位置设置到仿真器里边
    for name, value in initial_qpos.items():
        sim.data.set_joint_qpos(name, value)

    #步进仿真step函数的子函数，使上边的赋值在仿真器生效
    sim.forward()

    print('mujoco:', RbtMj.fk())
    # print('pykdl:', RbtKdl.fk())

    print('mujoco:', RbtMj.coriolis_gravity())
    # print('pykdl:', RbtKdl.coriolis() + RbtKdl.gravity_torque())

    while 1:
        # sim.data.ctrl[:] = RbtKdl.coriolis() + RbtKdl.gravity_torque()# sim.data.ctrl[:]输入力矩的值
        sim.data.ctrl[:] = RbtMj.coriolis_gravity()# sim.data.ctrl[:]输入力矩的值
        sim.step()
        viewer.render()


