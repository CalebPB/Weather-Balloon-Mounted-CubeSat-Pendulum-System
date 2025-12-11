import wind_data as wd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -------------------------------------------------------
# Parameters (Baseline)
# -------------------------------------------------------
g = 9.81
c_coefficient = 0.001075        # https://gibbs.ccny.cuny.edu/teaching/s2021/labs/GravitationalAcceleration/JAMP_2017012515591136.pdf?
rhospan = wd.rhospan            # Air density profile from wind_data.py
v_ascent = 5                    # CHANGE sondehub.org/balloon/ascentspeed

# Geometry
L2 = 2
Lpvc = 0.6
L5 = Lpvc / 2
phi = np.arctan(L2 / L5)
R3 = L5 / np.cos(phi)

# Diameters
d_scoop = 0.3048            # One foot diameter scoop
d_container = 0.1524        # 6” diameter container
t_container = 0.0030        # Thickness of schedule 40 pvc pipe (?) - Craig
d_pvc = 0.0334              # Outer diameter of 1” pvc pipe, schedule 80
r_pvc = d_pvc / 2
t_pvc = 0.0045              # Thickness of 1” pvc pipe, schedule 80
s_sat = 0.05                # Pocket cube dimension, 5cm

# Altitude and wind data
xspan = wd.xspan
vspan = wd.vspan

# Masses
m_container = 0.5
m_scoop = 2
m_pipe = 0.25
m_cubesat = 0.25

# Drag coefficients and areas
mu = 1.789e-5 										                #Update to make dynamic viscosity profile
Re_container = (rhospan * v_ascent * d_container) / mu 			    #Reynolds Number Container
Re_scoop = (rhospan * v_ascent * d_scoop) / mu 				        #Reynolds Number Scoop
c_dcontainer = 24/Re_container * (1 + 0.15 * Re_container**0.687)	#Drag Coefficient container
c_dscoop = 24/Re_scoop * (1 + 0.15 * Re_scoop**0.687)   			#Drag Coefficient scoop 
c_dcontainer = 0.3
c_dscoop = 0.2
A_container = np.pi * (d_container / 2) ** 2
A_scoop = np.pi * (d_scoop / 2) ** 2

# Inertia terms
Iscoop = m_scoop * ((0.4 * ((d_scoop/2)**2)) + (((L5 + (d_scoop/2))**2)+(L2**2)))
Ipvc = (m_pipe / 12) * (3 * (r_pvc ** 2 - (r_pvc - 2 * t_pvc) ** 2) + Lpvc ** 2) + m_pipe * (L2 + r_pvc) ** 2
Icubesat = m_cubesat / 6 * (s_sat ** 2) + m_cubesat * (L2 + (d_container - t_container - s_sat / 2) ** 2)
Icontainer = 2 * m_container / 5 * ((d_container / 2) ** 2 - ((d_container - 2 * t_container) / 2) ** 2) + m_container * (L2 + d_container / 2) ** 2

# Wind and ascent drag
F_wind_container = 0.5 * c_dcontainer * rhospan * A_container * vspan ** 2
F_wind_scoop = 0.5 * c_dscoop * rhospan * A_scoop * vspan ** 2
F_ascent_container = 0.5 * c_dcontainer * rhospan * A_container * v_ascent ** 2
F_ascent_scoop = 0.5 * c_dscoop * rhospan * A_scoop * v_ascent ** 2

# Interpolators
F_drag_container = interp1d(xspan, F_wind_container, fill_value="extrapolate")
F_drag_scoop = interp1d(xspan, F_wind_scoop, fill_value="extrapolate")
F_ascent_container_interp = interp1d(xspan, F_ascent_container, fill_value="extrapolate")
F_ascent_scoop_interp = interp1d(xspan, F_ascent_scoop, fill_value="extrapolate")

# =====================================================================
# Helper: Compute windowed amplitude/angle statistics
# =====================================================================
def compute_envelope_statistics(t, theta, window_size=50):
    n_windows = len(t) // window_size
    t_binned, mean_vals, max_vals, min_vals, std_vals = [], [], [], [], []
    for i in range(n_windows):
        start, end = i * window_size, (i + 1) * window_size
        segment = theta[start:end]
        t_binned.append(np.mean(t[start:end]))
        mean_vals.append(np.mean(segment))
        max_vals.append(np.max(segment))
        min_vals.append(np.min(segment))
        std_vals.append(np.std(segment))
    return (np.array(t_binned), np.array(mean_vals),
            np.array(max_vals), np.array(min_vals), np.array(std_vals))

# =====================================================================
# Equations of Motion (all geometry passed explicitly)
# =====================================================================
def payloadEOM(t, Y, F_drag_container, F_drag_scoop, F_ascent_scoop,
               I_local, Lpvc_local, L5, phi, R3, d_scoop, m_local_scoop):
    theta, dtheta = Y
    z = min(v_ascent * t, xspan[-1])
    Fd_container = float(F_drag_container(z))
    Fd_scoop = float(F_drag_scoop(z))
    Fa_container = float(F_ascent_container_interp(z))
    Fa_scoop = float(F_ascent_scoop(z))

    # Damping
    b = 2 * c_coefficient * I_local 

    # Geometry relations
    L3 = R3 * np.cos(phi + theta)
    L4 = 2 * L5 * np.cos(theta) - L3
    L6 = (L2 + d_container) * np.cos(theta)
    L7 = L2 / np.cos(theta) + (L5 + d_scoop / 2 - L2 * np.tan(theta)) * np.sin(theta)
    L8 = (L2 + d_container) * np.sin(theta)
    L9 = L7 - (Lpvc_local + 2 * (d_scoop / 2)) * np.sin(theta)

    M_restoring = ((m_pipe * g * L2 * np.sin(theta)) 
                   + (m_container * g * (L2 + d_container / 2) * np.sin(theta)) 
                   + (m_cubesat * g * (L2 + d_container - t_container - s_sat / 2)*np.sin(theta)) 
                   - (m_local_scoop * g * ((((L5 + d_scoop / 2)**2)+(L2**2))**0.5)*np.sin(np.pi/2 - phi - theta)) 
                   + (m_local_scoop * g * ((((L5 + d_scoop / 2)**2)+(L2**2))**0.5)*np.sin(np.pi/2 - phi + theta)))

    # Dynamics
    ddtheta = (-b * dtheta
               - M_restoring        # already including signs??? so + here
               - Fa_scoop * L4 
               + Fa_scoop * L3
               - Fa_container * L6
               + Fd_scoop * (L7 + L9)
               + Fd_container * L8) / I_local
    
    
    return [dtheta, ddtheta]

# =====================================================================
# Solve motion (passes all geometric parameters explicitly)
# =====================================================================
def solve_motion(theta0, F_drag_scoop, F_ascent_scoop,
                 I_local, Lpvc_local, L5, phi, R3, d_scoop, m_local_scoop, xmax=xspan[50]):
    Y0 = [theta0, 0]
    tmax = xmax / v_ascent
    sol = solve_ivp(
        payloadEOM,
        [0, tmax],
        Y0,
        # max_step=0.1,
        # rtol=1e-8,
        # atol=1e-8,
        args=(F_drag_container, F_drag_scoop, F_ascent_scoop,
              I_local, Lpvc_local, L5, phi, R3, d_scoop, m_local_scoop)
    )
    return sol.t, sol.y.T

# =====================================================================
# Normalize angle
# =====================================================================
def normalize(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

# =====================================================================
# Parameter Effects (Lpvc, d_scoop, m_scoop)
# =====================================================================
def compare_parameter_effects():
    Lpvc_values = [1.2, 0.6, 0.3]
    scoop_diameters = [0.4572, 0.3048, 0.1524]
    scoop_masses = [5, 2, 0.15]
    param_sets = [
        ("PVC Length Lpvc", Lpvc_values, "m"),
        ("Scoop Diameter d_scoop", scoop_diameters, "m"),
        ("Scoop Mass m_scoop", scoop_masses, "kg"),
    ]
    window_size = 100

    # --- Simple style mapping ---
    colors = ["blue", "red", "green"]
    line_widths = [2.5, 2.0, 1.5]

    for title, values, units in param_sets:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        for i, val in enumerate(values):
            # Default geometry copies
            Lpvc_local, d_scoop_local, m_scoop_local = Lpvc, d_scoop, m_scoop

            # --- Update selected parameter ---
            if "Lpvc" in title:
                Lpvc_local = val
            elif "d_scoop" in title:
                d_scoop_local = val
            elif "m_scoop" in title:
                m_scoop_local = val

            # --- Dependent geometry ---
            L5 = Lpvc_local / 2
            phi = np.arctan(L2 / L5)
            R3 = L5 / np.cos(phi)

            # --- Update area & forces ---
            A_scoop_local = np.pi * (d_scoop_local / 2) ** 2
            F_wind_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * vspan ** 2
            F_ascent_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * v_ascent ** 2
            F_drag_scoop_local = interp1d(xspan, F_wind_scoop, fill_value="extrapolate")
            F_ascent_scoop_local = interp1d(xspan, F_ascent_scoop, fill_value="extrapolate")

            # --- Update inertia ---
            Iscoop_local = m_scoop_local * (
                (0.4 * ((d_scoop_local / 2) ** 2))
                + (((Lpvc_local / 2 + (d_scoop_local / 2)) ** 2) + (L2 ** 2))
            )
            Ipvc_local = (m_pipe / 12) * (
                3 * (r_pvc ** 2 - (r_pvc - 2 * t_pvc) ** 2) + Lpvc_local ** 2
            ) + m_pipe * (L2 + r_pvc) ** 2
            I_local = Iscoop_local + Ipvc_local + Icubesat + Icontainer

            # --- Solve ---
            t, Y = solve_motion(
                np.radians(10),
                F_drag_scoop_local,
                F_ascent_scoop_local,
                I_local,
                Lpvc_local,
                L5,
                phi,
                R3,
                d_scoop_local,
                m_scoop_local,
            )
            theta = normalize(Y[:, 0])
            theta_deg = np.degrees(theta)
            t_bin, mean, max_val, min_val, std = compute_envelope_statistics(
                t, theta_deg, window_size
            )

            # --- Convert to altitude ---
            alt_bin = v_ascent * np.array(t_bin)  # <<<<<<<<<<<<<< ADDED

            # --- Label ---
            if "Lpvc" in title:
                label = f"$L_{{pvc}}$ = {val}{units}"
            elif "d_scoop" in title:
                label = f"$d_{{scoop}}$ = {val}{units}"
            elif "m_scoop" in title:
                label = f"$m_{{scoop}}$ = {val}{units}"

            color = colors[i]
            lw = line_widths[i]

            # --- Plot ---
            ax1.plot(alt_bin, mean, color=color, lw=lw, label=label)
            ax2.plot(alt_bin, (max_val - min_val) / 2, color=color, lw=lw, label=label)

        # --- Formatting ---
        ax1.set_xlabel("Altitude (m)")   # <<<<<<<<<<<<<< CHANGED
        ax1.set_ylabel("Angle (°)")
        ax1.legend()

        ax2.set_xlabel("Altitude (m)")   # <<<<<<<<<<<<<< CHANGED
        ax2.set_ylabel("Amplitude (°)")
        ax2.legend()

        plt.tight_layout()
        plt.show()

# =====================================================================
# Ideal Setup Simulation
# =====================================================================
def ideal_setup_plots():
    ideal_Lpvc = 0.6
    ideal_d_scoop = 0.3048
    ideal_m_scoop = 2
    print("\nGenerating ideal setup plot (Theta vs Distance)...")

    # --- Geometry ---
    L5 = ideal_Lpvc / 2
    phi = np.arctan(L2 / L5)
    R3 = L5 / np.cos(phi)

    # --- Aerodynamics ---
    A_scoop_local = np.pi * (ideal_d_scoop / 2) ** 2
    F_wind_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * vspan ** 2
    F_ascent_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * v_ascent ** 2
    F_drag_scoop_local = interp1d(xspan, F_wind_scoop, fill_value="extrapolate")
    F_ascent_scoop_local = interp1d(xspan, F_ascent_scoop, fill_value="extrapolate")

    # --- Inertia ---
    Iscoop_local = ideal_m_scoop * (
        0.4 * (ideal_d_scoop / 2)**2 +
        ((ideal_Lpvc / 2 + ideal_d_scoop / 2)**2 + L2**2)
    )
    Ipvc_local = (m_pipe / 12) * (
        3 * (r_pvc**2 - (r_pvc - 2*t_pvc)**2) + ideal_Lpvc**2
    ) + m_pipe * (L2 + r_pvc)**2

    I_local = Iscoop_local + Ipvc_local + Icubesat + Icontainer

    # --- Solve EOM ---
    t, Y = solve_motion(
        np.radians(10),
        F_drag_scoop_local,
        F_ascent_scoop_local,
        I_local,
        ideal_Lpvc,
        L5,
        phi,
        R3,
        ideal_d_scoop,
        ideal_m_scoop,
    )
    theta_deg = np.degrees(normalize(Y[:, 0]))

    # --- Convert time to distance ---
    distance = v_ascent * t  # x = v*t

    # --- Plot θ vs distance ---
    plt.figure(figsize=(10, 5))
    plt.plot(distance, theta_deg, color='k', lw=1.8)
    plt.title("Ideal Configuration: Transient Response θ vs Distance")
    plt.xlabel("Distance Traveled (m)")
    plt.ylabel("Theta (°)")
    plt.tight_layout()
    plt.show()


def heatmap_mass_diameter(
        mass_range=np.linspace(0.5, 8, 12),
        diameter_range=np.linspace(0.15, 0.50, 12),
        theta0_deg=10,              # <<--- NEW
        window_size=100,
        metric="amplitude"
    ):
    """
    Generates a heatmap of oscillation performance vs scoop mass & diameter.
    Includes control of initial deflection angle theta0_deg.
    """

    Z = np.zeros((len(mass_range), len(diameter_range)))
    theta0 = np.radians(theta0_deg)   # <<--- convert once

    for i, m_local in enumerate(mass_range):
        for j, d_local in enumerate(diameter_range):

            # --- Geometry ---
            Lpvc_local = Lpvc
            L5_local = Lpvc_local / 2
            phi_local = np.arctan(L2 / L5_local)
            R3_local = L5_local / np.cos(phi_local)

            # --- Aerodynamics ---
            A_scoop_local = np.pi * (d_local / 2)**2
            F_wind_scoop_local = 0.5 * c_dscoop * rhospan * A_scoop_local * vspan**2
            F_ascent_scoop_local = 0.5 * c_dscoop * rhospan * A_scoop_local * v_ascent**2
            F_drag_scoop_local = interp1d(xspan, F_wind_scoop_local, fill_value="extrapolate")
            F_ascent_scoop_local = interp1d(xspan, F_ascent_scoop_local, fill_value="extrapolate")

            # --- Inertia ---
            Iscoop_local = m_local * (
                0.4*(d_local/2)**2 + ((L5_local + d_local/2)**2 + L2**2)
            )
            Ipvc_local = (m_pipe/12)*(3*(r_pvc**2 - (r_pvc-2*t_pvc)**2)
                           + Lpvc_local**2) + m_pipe*(L2+r_pvc)**2

            I_local = Iscoop_local + Ipvc_local + Icubesat + Icontainer

            # --- Solve ---
            t, Y = solve_motion(
                theta0,                 # <<--- uses your initial theta
                F_drag_scoop_local,
                F_ascent_scoop_local,
                I_local,
                Lpvc_local,
                L5_local,
                phi_local,
                R3_local,
                d_local,
                m_local
            )

            theta_deg = np.degrees(normalize(Y[:, 0]))
            t_bin, mean, max_val, min_val, std = compute_envelope_statistics(
                t, theta_deg, window_size
            )

            # --- Select performance metric ---
            if metric == "amplitude":
                value = (max_val[-1] - min_val[-1]) / 2
            elif metric == "std":
                value = std[-1]
            elif metric == "mean":
                value = mean[-1]
            else:
                raise ValueError("Unknown metric")

            Z[i, j] = value

    # --- Plot ---
    plt.figure(figsize=(10, 7))
    plt.imshow(Z, origin='lower', aspect='auto',
               extent=[diameter_range[0], diameter_range[-1],
                       mass_range[0], mass_range[-1]])
    plt.colorbar(label=f"Final {metric} (deg)")
    plt.xlabel("Scoop Diameter (m)")
    plt.ylabel("Scoop Mass (kg)")
    plt.title(f"Mass–Diameter Heatmap (theta0={theta0_deg}°)")
    plt.tight_layout()
    plt.show()

    return Z


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    theta0 = np.radians(15)
    # compare_parameter_effects()
    ideal_setup_plots()
#     heatmap_mass_diameter(
#     mass_range=np.linspace(0.15, 5, 10),
#     diameter_range=np.linspace(0.15, 0.45, 10),
#     theta0_deg=15,         # <-- starting angle
#     metric="mean"
# )
