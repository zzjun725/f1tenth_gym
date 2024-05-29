import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import math


############## Dynamic Models for [steer, velocity] control input ####################
@njit(cache=True)
def steer_pid(current_steer, desired_steer, sv_min, sv_max, dt):
    steer_diff = desired_steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        steer_v = steer_diff / dt
    else:
        steer_v = 0.0
    steer_v = max(sv_min, min(sv_max, steer_v))
    return steer_v


@njit(cache=True)
def accel_pid(current_speed, desired_speed, switch_v, min_accel, max_accel, dt):
    vel_diff = desired_speed - current_speed
    accel = vel_diff / dt
    accel = min(max_accel, max(min_accel, accel))
    if current_speed > switch_v:
        accel = accel * switch_v / current_speed
    return accel


@njit(cache=True)
def njit_kinematic_st(x, u, lf, lr, sv_min, sv_max, steer_delay_time, a_min, a_max, accel_delay_time):
    sv = steer_pid(x[2], u[0], sv_min, sv_max, steer_delay_time)
    accel = accel_pid(x[4], u[1], a_min, a_max, accel_delay_time)

    lwb = lf + lr
    f = np.array([x[4] * np.cos(x[3]),
                  x[4] * np.sin(x[3]),
                  sv,
                  x[4] / lwb * np.tan(x[2]),
                  accel])
    return f


def kinematic_st(x, u, p):
    return njit_kinematic_st(x, u, p["lf"], p["lr"], p["sv_min"], p["sv_max"], p["steer_delay_time"], p["a_min"],
                             p["a_max"], p["accel_delay_time"])


@njit(cache=True)
def njit_dynamic_st_pacejka(x, u, lf, lr, h, m, I, steer_delay_time, g, mu, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r,
                            sv_min, sv_max, v_switch, a_min, a_max, accel_delay_time):
    sv = steer_pid(x[2], u[0], sv_min, sv_max, steer_delay_time)
    accel = accel_pid(x[4], u[1], v_switch, a_min, a_max, accel_delay_time)

    lwb = lf + lr
    if abs(x[4]) < 1.0:
        # Simplified low-speed kinematic model
        f_ks = np.array([x[4] * np.cos(x[3]),
                         x[4] * np.sin(x[3]),
                         sv,
                         x[4] / lwb * np.tan(x[2]),
                         accel])
        f = np.hstack((f_ks, np.array([u[1] / lwb * np.tan(x[2]) + x[4] / (lwb * np.cos(x[2]) ** 2) * u[0], 0])))
        return f

    # Compute slip angles and vertical tire forces
    alpha_f = -math.atan((x[5] + x[6] * lf) / x[4]) + x[2]
    alpha_r = -math.atan((x[5] - x[6] * lr) / x[4])
    F_zf = m * (-u[1] * h + g * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g * lf) / (lr + lf)

    # Calculate lateral forces using Pacejka's magic formula
    F_yf = mu * F_zf * D_f * math.sin(C_f * math.atan(B_f * alpha_f - E_f * (B_f * alpha_f - math.atan(B_f * alpha_f))))
    F_yr = mu * F_zr * D_r * math.sin(C_r * math.atan(B_r * alpha_r - E_r * (B_r * alpha_r - math.atan(B_r * alpha_r))))

    f = np.array([x[4] * np.cos(x[3]) - x[5] * math.sin(x[3]),
                  x[4] * np.sin(x[3]) + x[5] * math.cos(x[3]),
                  sv,
                  x[6],
                  accel,
                  1 / m * (F_yr + F_yf) - x[4] * x[6],
                  1 / I * (-lr * F_yr + lf * F_yf * math.cos(x[2]))])
    return f


@njit(cache=True)
def njit_dynamic_st_linear(x, u, lf, lr, h, m, I, steer_delay_time, g, mu, C_Sf, C_Sr, sv_min, sv_max, v_switch, a_min,
                           a_max, accel_delay_time):
    sv = steer_pid(x[2], u[0], sv_min, sv_max, steer_delay_time)
    accel = accel_pid(x[4], u[1], v_switch, a_min, a_max, accel_delay_time)

    lwb = lf + lr
    if abs(x[4]) < 1.0:
        # Simplified low-speed kinematic model
        f_ks = np.array([x[4] * np.cos(x[3]),
                         x[4] * np.sin(x[3]),
                         sv,
                         x[4] / lwb * np.tan(x[2]),
                         accel])
        f = np.hstack((f_ks, np.array([u[1] / lwb * np.tan(x[2]) + x[4] / (lwb * np.cos(x[2]) ** 2) * u[0], 0])))
        return f

    # Compute slip angles and vertical tire forces
    alpha_f = -math.atan((x[5] + x[6] * lf) / x[4]) + x[2]
    alpha_r = -math.atan((x[5] - x[6] * lr) / x[4])
    F_zf = m * (-u[1] * h + g * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g * lf) / (lr + lf)

    # Calculate lateral forces using linear tire model
    F_yf = mu * F_zf * C_Sf * alpha_f
    F_yr = mu * F_zr * C_Sr * alpha_r

    f = np.array([x[4] * np.cos(x[3]) - x[5] * math.sin(x[3]),
                  x[4] * np.sin(x[3]) + x[5] * math.cos(x[3]),
                  sv,
                  x[6],
                  accel,
                  1 / m * (F_yr + F_yf) - x[4] * x[6],
                  1 / I * (-lr * F_yr + lf * F_yf * math.cos(x[2]))])
    return f


# Wrapper functions that prepare parameters and choose the correct JIT function
def dynamic_st(x, u, p, tire_type):
    if tire_type == "pacejka":
        return njit_dynamic_st_pacejka(
            x, u, p["lf"], p["lr"], p["h"], p["m"], p["I"], p["steer_delay_time"], 9.81, p["mu"],
            p["B_f"], p["C_f"], p["D_f"], p["E_f"], p["B_r"], p["C_r"], p["D_r"], p["E_r"],
            p["sv_min"], p["sv_max"], p["v_switch"], p["a_min"], p["a_max"], p["accel_delay_time"]
        )
    elif tire_type == "linear":
        return njit_dynamic_st_linear(
            x, u, p["lf"], p["lr"], p["h"], p["m"], p["I"], p["steer_delay_time"], 9.81, p["mu"],
            p["C_Sf"], p["C_Sr"], p["sv_min"], p["sv_max"], p["v_switch"], p["a_min"], p["a_max"], p["accel_delay_time"]
        )


############## Dynamic Models for [steer, velocity] control input ####################

dynamic_f110_idx = {
    'X': 0, 'Y': 1, 'DELTA': 2, 'YAW': 3, 'VX': 4, 'VY': 5, 'YAWRATE': 6
}


dynamic_idx = {
    'X': 0, 'Y': 1, 'DELTA': 2, 'VX': 3, 'YAW': 4, 'YAWRATE': 5, 'BETA': 6
}