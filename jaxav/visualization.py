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

    # for i, pos in state.pos:
    #     ...
    
    def update(state: DroneState):
        transform = Transform(pos=state.pos, rot=state.rot)
        arms = transform(np.stack([np.zeros_like(state.rotor_trans), state.rotor_trans], -2))
        for i, drone_i_arms in enumerate(arms):
            lines = drone_arms[i]
            for line, arm in zip(lines, drone_i_arms):
                line.set_data(*arm.T[:2])
                line.set_3d_properties(arm.T[2])
    
    anim = animation.FuncAnimation(
        fig, update, states[1:]
    )
    return anim

