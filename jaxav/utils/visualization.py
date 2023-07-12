import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jaxav.dynamics import DroneState, Transform
from typing import Sequence

def render(states: Sequence[DroneState]):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 4)
    ax.plot([0, 1], [0, 0], [0, 0])

    state = states[0]
    
    drone_arms = {}
    trajs = {}
    transform = Transform(pos=state.pos, rot=state.rot)
    
    arms = np.stack([np.zeros_like(state.rotor_trans), state.rotor_trans], -2)
    arms = transform(arms)

    for i, drone_i_arms in enumerate(arms):
        lines = []
        for arm in drone_i_arms:
            line, = ax.plot(*arm.T)
            lines.append(line)
        drone_arms[i] = lines

    for i, pos in enumerate(state.pos):
        trajs[i] = (ax.plot(*pos.T)[0], [pos])
    
    def update(state: DroneState):
        transform = Transform(pos=state.pos, rot=state.rot)
        arms = transform(np.stack([np.zeros_like(state.rotor_trans), state.rotor_trans], -2))
        for i, drone_i_arms in enumerate(arms):
            arm_lines = drone_arms[i]
            for arm_line, arm in zip(arm_lines, drone_i_arms):
                arm_line.set_data(*arm.T[:2])
                arm_line.set_3d_properties(arm.T[2])
            traj_line, traj = trajs[i]
            traj.append(state.pos[i])
            xyz = np.array(traj)
            traj_line.set_data(*xyz.T[:2])
            traj_line.set_3d_properties(xyz.T[2])
    
    anim = animation.FuncAnimation(
        fig, update, states[1:]
    )
    return anim


class Traj3D:
    """
    Helper class for visualizing a trajectory in matplotlib animation.
    """
    def __init__(self, ax, x0):
        self.traj = [x0]
        self.line = ax.plot(*x0.T)[0]
    
    def update(self, x):
        self.traj.append(x)
        x, y, z = zip(*self.traj)
        self.line.set_data(x, y)
        self.line.set_3d_properties(z)
        return self.line


class Drone:
    """
    Helper class for visualizing a multirotor in matplotlib animation.
    """
    def __init__(self, ax, s0: DroneState):
        self.arm_lines = [ax.plot(*arm.T, "--")[0] for arm in self._arms(s0)]
        heading = np.stack([s0.pos, s0.pos+s0.heading], axis=-2)
        self.heading_line = ax.plot(*heading.T, "--")[0]

    def _arms(self, state: DroneState):
        transform = Transform(state.pos, state.rot)
        arms = np.stack([
            np.zeros_like(state.rotor_trans), 
            state.rotor_trans
        ], axis=-2)
        arms = transform(arms)
        return arms
    
    def update(self, state: DroneState):
        for arm_line, arm in zip(self.arm_lines, self._arms(state)):
            arm_line.set_data(*arm.T[:2])
            arm_line.set_3d_properties(arm.T[2])
        heading = np.stack([state.pos, state.pos+state.heading], axis=-2)
        self.heading_line.set_data(*heading.T[:2])
        self.heading_line.set_3d_properties(heading.T[2])
        return self.arm_lines

