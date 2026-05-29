import numpy as np

def initialize_state(formation, baseline, att_F, m_prop_init_F, x_init_L, n_sc):
    
    # Leader is always the first in the stack
    x0 = [x_init_L]
    
    # Initial follower conditions
    q_init_F = np.array([1.0, 0.0, 0.0, 0.0])
    w_init_F = np.array([0.001, 0.01, 0.0])
    
    n_followers = n_sc - 1
    delta_r0_list = []

    if att_F != "same as leader":
        raise NotImplementedError("Attitude initialization other than 'same as leader' not implemented.")

    # Explicit handling for the 5 spacecraft (4 followers) cross/square formation
    if n_sc == 5 and formation == "square planar":
        delta_r0_list = [
            np.array([ baseline,  0.0,      0.0]),   # follower_1: +x
            np.array([-baseline,  0.0,      0.0]),   # follower_2: -x
            np.array([ 0.0,       baseline, 0.0]),   # follower_3: +y
            np.array([ 0.0,      -baseline, 0.0]),   # follower_4: -y
        ]
        
    # Generalized N-gon handling for all other N (or if formation != 'square planar')
    else:
        for i in range(n_followers):
            # Calculate the angle around the circle for this specific follower
            theta = 2.0 * np.pi * i / n_followers
            
            # Convert polar coordinates (baseline, theta) to Cartesian (x, y)
            pos_x = baseline * np.cos(theta)
            pos_y = baseline * np.sin(theta)
            pos_z = 0.0
            
            delta_r0_list.append(np.array([pos_x, pos_y, pos_z]))

    # Stack the states
    for dr0 in delta_r0_list:
        dv0 = np.zeros(3)
        x0.append(np.concatenate([dr0, dv0, q_init_F, w_init_F, [m_prop_init_F]]))
        
    x = np.concatenate(x0)
    return x