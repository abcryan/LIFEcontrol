import numpy as np
import matplotlib.pyplot as plt


def plot_controller_summary(
    t_hist, X_hist, U_hist, YREF_hist,
    dim_x_sc=14, dim_u_sc=20, follower_idx=1,
    T_max=0.003, B_F=None,
):
    """Summary diagnostics for the translation PID on follower `follower_idx`.

    Expects t_hist [s], X_hist (N+1, dim_x), U_hist (N+1, dim_u),
    YREF_hist (N+1, dim_y). Pass B_F (3x20) if you want the commanded body
    force overlay on the thrust panel.
    """
    ix = dim_x_sc * follower_idx
    iu = dim_u_sc * follower_idx

    t_h   = t_hist / 3600.0
    dr    = X_hist[:, ix:ix+3]      * 1e3      # km → m
    dv    = X_hist[:, ix+3:ix+6]    * 1e3      # km/s → m/s
    drR   = YREF_hist[:, ix:ix+3]   * 1e3
    dvR   = YREF_hist[:, ix+3:ix+6] * 1e3
    q     = X_hist[:, ix+6:ix+10]
    w     = X_hist[:, ix+10:ix+13]
    mprop = X_hist[:, ix+13]
    T     = U_hist[:, iu:iu+dim_u_sc]          # (N+1, 20)

    err_r = np.linalg.norm(dr - drR, axis=1)
    err_v = np.linalg.norm(dv - dvR, axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(f'Controller summary — follower {follower_idx}', fontsize=14)

    colors = ['C0', 'C1', 'C2']
    labels = ['x', 'y', 'z']

    # (0,0) δr tracking
    ax = axes[0, 0]
    for k in range(3):
        ax.plot(t_h, dr[:, k],  color=colors[k], label=f'δr_{labels[k]}')
        ax.plot(t_h, drR[:, k], color=colors[k], linestyle='--', alpha=0.6)
    ax.set_xlabel('t [h]'); ax.set_ylabel('δr [m]')
    ax.set_title('Position tracking (solid = actual, dashed = ref)')
    ax.grid(True); ax.legend(loc='upper right', fontsize=8)

    # (0,1) δv tracking
    ax = axes[0, 1]
    for k in range(3):
        ax.plot(t_h, dv[:, k],  color=colors[k], label=f'δv_{labels[k]}')
        ax.plot(t_h, dvR[:, k], color=colors[k], linestyle='--', alpha=0.6)
    ax.set_xlabel('t [h]'); ax.set_ylabel('δv [m/s]')
    ax.set_title('Velocity tracking')
    ax.grid(True); ax.legend(loc='upper right', fontsize=8)

    # (0,2) XY trajectory
    ax = axes[0, 2]
    ax.plot(drR[:, 0], drR[:, 1], 'k--', alpha=0.5, label='reference')
    ax.plot(dr[:, 0],  dr[:, 1],  'C0', label='actual')
    ax.scatter([dr[0, 0]],  [dr[0, 1]],  c='g', marker='o', s=40, zorder=5, label='start')
    ax.scatter([dr[-1, 0]], [dr[-1, 1]], c='r', marker='x', s=50, zorder=5, label='end')
    ax.scatter([0], [0], c='k', marker='+', s=80, label='leader')
    ax.set_xlabel('δr_x [m]'); ax.set_ylabel('δr_y [m]')
    ax.set_title('XY trajectory'); ax.set_aspect('equal')
    ax.grid(True); ax.legend(loc='upper right', fontsize=8)

    # (1,0) tracking errors
    ax = axes[1, 0]
    ax.plot(t_h, err_r, 'C3', label='|δr − δr_ref|')
    ax.set_xlabel('t [h]'); ax.set_ylabel('position error [m]', color='C3')
    ax.tick_params(axis='y', labelcolor='C3')
    ax.set_title('Tracking error magnitudes')
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(t_h, err_v, 'C4', label='|δv − δv_ref|')
    ax2.set_ylabel('velocity error [m/s]', color='C4')
    ax2.tick_params(axis='y', labelcolor='C4')

    # (1,1) all 20 thrusters + cap
    ax = axes[1, 1]
    for l in range(dim_u_sc):
        ax.plot(t_h, T[:, l], lw=0.7, alpha=0.7)
    ax.axhline(T_max, color='k', linestyle='--', lw=1.5, label=f'T_max = {T_max} N')
    ax.set_xlabel('t [h]'); ax.set_ylabel('T_l [N]')
    ax.set_title(f'Thruster magnitudes (all {dim_u_sc})')
    ax.grid(True); ax.legend(loc='upper right', fontsize=8)

    # (1,2) total thrust + commanded body force
    ax = axes[1, 2]
    T_sum = T.sum(axis=1)
    ax.plot(t_h, T_sum, 'C0', label='Σ T_l (sum of thrusts)')
    if B_F is not None:
        F_body_realized = -(B_F @ T.T).T            # (N+1, 3)
        F_mag = np.linalg.norm(F_body_realized, axis=1)
        ax.plot(t_h, F_mag, 'C2', label='|F^B| realized')
    ax.axhline(T_max * dim_u_sc, color='k', linestyle=':', lw=1,
               label=f'theoretical Σ-cap = {T_max*dim_u_sc:.3f} N')
    ax.set_xlabel('t [h]'); ax.set_ylabel('[N]')
    ax.set_title('Net actuation')
    ax.grid(True); ax.legend(loc='upper right', fontsize=8)

    # (2,0) propellant
    ax = axes[2, 0]
    ax.plot(t_h, mprop, 'C5')
    ax.set_xlabel('t [h]'); ax.set_ylabel('m_prop [kg]')
    used = mprop[0] - mprop[-1]
    ax.set_title(f'Propellant (used: {used*1e3:.3f} g)')
    ax.grid(True)

    # (2,1) quaternion norm + angular rate norm
    ax = axes[2, 1]
    qnorm_err = np.abs(np.linalg.norm(q, axis=1) - 1.0)
    ax.semilogy(t_h, qnorm_err + 1e-16, 'C6', label='||q| − 1|')
    ax.set_xlabel('t [h]'); ax.set_ylabel('|q|−1', color='C6')
    ax.tick_params(axis='y', labelcolor='C6')
    ax.set_title('Attitude sanity')
    ax.grid(True, which='both')
    ax2 = ax.twinx()
    ax2.plot(t_h, np.linalg.norm(w, axis=1), 'C7', label='|ω|')
    ax2.set_ylabel('|ω| [rad/s]', color='C7')
    ax2.tick_params(axis='y', labelcolor='C7')

    # (2,2) thruster duty cycle / saturation summary
    ax = axes[2, 2]
    duty = (T > 0.99 * T_max).mean(axis=0) * 100.0   # % time saturated, per thruster
    active = (T > 1e-6).mean(axis=0) * 100.0
    idx = np.arange(dim_u_sc)
    ax.bar(idx - 0.2, active, width=0.4, label='% time active',     color='C0')
    ax.bar(idx + 0.2, duty,   width=0.4, label='% time saturated',  color='C3')
    ax.set_xlabel('thruster index'); ax.set_ylabel('% of run')
    ax.set_title('Thruster usage')
    ax.set_xticks(idx); ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, axis='y'); ax.legend(loc='upper right', fontsize=8)

    # summary metrics — printed text box
    rms_r = np.sqrt(np.mean(err_r**2))
    max_r = err_r.max()
    rms_v = np.sqrt(np.mean(err_v**2))
    sat_any = (T.max(axis=1) > 0.99 * T_max).mean() * 100.0
    txt = (
        f'RMS position error: {rms_r*1e3:.3f} mm\n'
        f'Max position error: {max_r*1e3:.3f} mm\n'
        f'RMS velocity error: {rms_v*1e3:.4f} mm/s\n'
        f'Saturation (any thruster): {sat_any:.1f}% of time\n'
        f'Propellant used:    {used*1e3:.3f} g'
    )
    fig.text(0.01, 0.01, txt, fontsize=9, family='monospace',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()