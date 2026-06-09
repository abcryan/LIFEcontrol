import numpy as np
import matplotlib.pyplot as plt

def test_guidance_trajectory(guidance, dim_x_sc=14, dt=100.0, t_tot=86400.0):
    """Sanity-check the guidance reference by plotting δr and δv vs time
    for the first follower spacecraft. No dynamics, no control — just the
    raw reference signal coming out of Guidance.reference().
    """

    n_steps = int(t_tot / dt)
    t       = np.zeros(n_steps)
    dr      = np.zeros((n_steps, 3))
    dv      = np.zeros((n_steps, 3))

    # follower-1 block start index
    i = dim_x_sc

    et_dummy = 0.0   # not used inside reference() right now
    for k in range(n_steps):
        y_ref     = guidance.reference(et_dummy, dt)
        dr[k, :]  = y_ref[i+0 : i+3]
        dv[k, :]  = y_ref[i+3 : i+6]
        t[k]      = k * dt

    # reset internal clock so subsequent sim use starts clean
    guidance.t = 0.0

    t_h = t / 3600.0   # hours, just for readability

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # δr components vs time
    ax = axes[0, 0]
    ax.plot(t_h, dr[:, 0], label='δr_x')
    ax.plot(t_h, dr[:, 1], label='δr_y')
    ax.plot(t_h, dr[:, 2], label='δr_z')
    ax.set_xlabel('time [h]'); ax.set_ylabel('δr [state units]')
    ax.set_title('Follower-1 reference position')
    ax.grid(True); ax.legend()

    # δv components vs time
    ax = axes[0, 1]
    ax.plot(t_h, dv[:, 0], label='δv_x')
    ax.plot(t_h, dv[:, 1], label='δv_y')
    ax.plot(t_h, dv[:, 2], label='δv_z')
    ax.set_xlabel('time [h]'); ax.set_ylabel('δv [state units / s]')
    ax.set_title('Follower-1 reference velocity')
    ax.grid(True); ax.legend()

    # XY trace of δr (should be a circle in XY)
    ax = axes[1, 0]
    ax.plot(dr[:, 0], dr[:, 1])
    ax.scatter([0], [0], c='k', marker='+', label='leader')
    ax.set_xlabel('δr_x'); ax.set_ylabel('δr_y')
    ax.set_title('Follower-1 reference orbit (XY)')
    ax.set_aspect('equal'); ax.grid(True); ax.legend()

    # |δr| and |δv| vs time — should be flat for a perfect circle
    ax = axes[1, 1]
    ax.plot(t_h, np.linalg.norm(dr, axis=1), label='|δr|')
    ax.plot(t_h, np.linalg.norm(dv, axis=1), label='|δv|')
    ax.set_xlabel('time [h]'); ax.set_ylabel('magnitude')
    ax.set_title('Magnitudes (should be constant)')
    ax.grid(True); ax.legend()

    fig.tight_layout()
    plt.show()