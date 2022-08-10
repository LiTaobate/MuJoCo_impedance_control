import numpy as np
import mujoco_py as mp


class RobotMj:
    def __init__(self, sim, ee):
        self.sim = sim
        self.ee_name = ee
        self.index = np.arange(0, 6)

    def fk(self):
        pos = self.sim.data.get_site_xpos(self.ee_name)
        ori = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.ee_name)].reshape([3, 3]))
        return pos, ori

    def jacobian(self):
        J_pos = np.array(self.sim.data.get_site_jacp(self.ee_name).reshape((3, -1))[:, self.index])
        J_ori = np.array(self.sim.data.get_site_jacr(self.ee_name).reshape((3, -1))[:, self.index])
        J_full = np.array(np.vstack([J_pos, J_ori]))
        return J_full

    def mass_matrix(self):#质量矩阵 来自rebosuite
        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mp.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        mass_matrix = mass_matrix[self.index, :][:, self.index]
        return mass_matrix

    def coriolis_gravity(self):#科氏力和重力的和C(q，q)+G(q)
        return self.sim.data.qfrc_bias[self.index]
