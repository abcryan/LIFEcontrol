import numpy as np

# Initialize the stacked state vector for leader + followers.
def initialize_state(formation, baseline, att_F, m_prop_init_F, x_init_L, n_sc):

    if formation == "square planar" and att_F == "same as leader" and n_sc == 5:

        delta_r0_list = [
            np.array([ baseline,  0.0,      0.0]),   # follower_1: +x
            np.array([-baseline,  0.0,      0.0]),   # follower_2: -x
            np.array([ 0.0,       baseline, 0.0]),   # follower_3: +y
            np.array([ 0.0,      -baseline, 0.0]),   # follower_4: -y
        ]

        q_init_F = np.array([1.0, 0.0, 0.0, 0.0])
        w_init_F = np.array([0.001, 0.01, 0.0])

        x0 = [x_init_L]
        for dr0 in delta_r0_list:
            dv0 = np.zeros(3)
            x0.append(np.concatenate([dr0, dv0, q_init_F, w_init_F, [m_prop_init_F]]))
        x = np.concatenate(x0)

    else:
        raise NotImplementedError(
            "Formation geometry or attitude initialization not implemented."
        )

    return x

