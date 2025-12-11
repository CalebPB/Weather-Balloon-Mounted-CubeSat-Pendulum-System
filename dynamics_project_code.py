import wind_data as wd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import pandas as pd

# -------------------------------------------------------
# Parameters (Baseline)
# -------------------------------------------------------
g = 9.81
c_coefficient = 0.001075
rhospan = wd.rhospan
v_ascent = 5

# Geometry
L2 = 2
Lpvc = 0.6
L5 = Lpvc / 2
phi = np.arctan(L2 / L5)
R3 = L5 / np.cos(phi)

# Diameters
d_scoop = 0.3048
d_container = 0.1524
t_container = 0.0030
d_pvc = 0.0334
r_pvc = d_pvc / 2
t_pvc = 0.0045
s_sat = 0.05

# Altitude and wind data
xspan = wd.xspan
vspan = wd.vspan

# Masses
m_container = 0.5
m_scoop = 2
m_pipe = 0.25
m_cubesat = 0.25

# Drag coefficients and areas
mu = 1.789e-5
Re_container = (rhospan * v_ascent * d_container) / mu
Re_scoop = (rhospan * v_ascent * d_scoop) / mu
c_dcontainer = 24/Re_container * (1 + 0.15 * Re_container**0.687)
c_dscoop = 24/Re_scoop * (1 + 0.15 * Re_scoop**0.687)
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
# Equations of Motion
# =====================================================================
def payloadEOM(t, Y, F_drag_container, F_drag_scoop, F_ascent_scoop,
               I_local, Lpvc_local, L5, phi, R3, d_scoop, m_local_scoop):
    theta, dtheta = Y
    z = min(v_ascent * t, xspan[-1])
    Fd_container = float(F_drag_container(z))
    Fd_scoop = float(F_drag_scoop(z))
    Fa_container = float(F_ascent_container_interp(z))
    Fa_scoop = float(F_ascent_scoop(z))

    b = 2 * c_coefficient * I_local 

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

    ddtheta = (-b * dtheta
               - M_restoring
               - Fa_scoop * L4 
               + Fa_scoop * L3
               - Fa_container * L6
               + Fd_scoop * (L7 + L9)
               + Fd_container * L8) / I_local
    
    return [dtheta, ddtheta]

# =====================================================================
# Solve motion
# =====================================================================
def solve_motion(theta0, F_drag_scoop, F_ascent_scoop,
                 I_local, Lpvc_local, L5, phi, R3, d_scoop, m_local_scoop, xmax=xspan[500]):
    Y0 = [theta0, 0]
    tmax = xmax / v_ascent
    sol = solve_ivp(
        payloadEOM,
        [0, tmax],
        Y0,
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
# 3x2 Grid: Publication-Quality Figure + Summary Table
# =====================================================================
def compare_parameter_effects_publication():
    """
    Creates publication-quality figures:
    1. 3x2 grid (Lpvc, d_scoop, and m_scoop) - all parameters
    2. Summary metrics table
    All units in metric (meters, kg, degrees)
    """
    
    # Parameter sets - ALL THREE (all in metric)
    pvc_lengths = [0.3, 0.6, 1.2]  # m
    scoop_diameters = [0.1524, 0.3048, 0.4572]  # m (6", 12", 18" converted)
    scoop_masses = [0.15, 2, 5]  # kg
    
    # Colors for each parameter
    colors_pvc = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    colors_scoop = ['#d62728', '#9467bd', '#8c564b']  # Red, Purple, Brown
    colors_mass = ['#e377c2', '#7f7f7f', '#bcbd22']   # Pink, Gray, Yellow-Green
    
    window_size = 100
    
    # Storage for metrics table
    metrics_data = []
    
    # Small helper to append metrics in a consistent format
    def add_metric(param_name, value_str, mean_arr, amplitude_arr):
        metrics_data.append({
            'Parameter': param_name,
            'Value': value_str,
            'Peak Mean (deg)': f'{np.max(mean_arr):.2f}',
            'Peak Amplitude (deg)': f'{np.max(amplitude_arr):.2f}',
            'Final Mean (deg)': f'{mean_arr[-1]:.2f}',
            'Final Amplitude (deg)': f'{amplitude_arr[-1]:.2f}'
        })
    
    # ====================================================================
    # FIGURE 1: Main 3x2 Publication Figure
    # ====================================================================
    
    # Use 3x2 layout - taller to accommodate 3 rows
    fig1 = plt.figure(figsize=(12, 15))
    
    # Create 3x2 grid
    gs = gridspec.GridSpec(3, 2, figure=fig1, 
                          hspace=0.35, wspace=0.3,
                          left=0.10, right=0.95, 
                          top=0.96, bottom=0.04)
    
    # Create axes
    ax_pvc_mean = fig1.add_subplot(gs[0, 0])
    ax_pvc_amp = fig1.add_subplot(gs[0, 1])
    ax_scoop_mean = fig1.add_subplot(gs[1, 0])
    ax_scoop_amp = fig1.add_subplot(gs[1, 1])
    ax_mass_mean = fig1.add_subplot(gs[2, 0])
    ax_mass_amp = fig1.add_subplot(gs[2, 1])
    
    # -----------------------------
    # ROW 1: PVC Length (Lpvc)
    # -----------------------------
    print("Computing PVC length effects...")
    for i, lpvc_val in enumerate(pvc_lengths):
        # Setup geometry
        Lpvc_local = lpvc_val
        d_scoop_local = d_scoop
        m_scoop_local = m_scoop
        
        L5 = Lpvc_local / 2
        phi = np.arctan(L2 / L5)
        R3 = L5 / np.cos(phi)
        
        # Update forces
        A_scoop_local = np.pi * (d_scoop_local / 2) ** 2
        F_wind_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * vspan ** 2
        F_ascent_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * v_ascent ** 2
        F_drag_scoop_local = interp1d(xspan, F_wind_scoop, fill_value="extrapolate")
        F_ascent_scoop_local = interp1d(xspan, F_ascent_scoop, fill_value="extrapolate")
        
        # Update inertia
        Iscoop_local = m_scoop_local * (
            (0.4 * ((d_scoop_local / 2) ** 2))
            + (((Lpvc_local / 2 + (d_scoop_local / 2)) ** 2) + (L2 ** 2))
        )
        Ipvc_local = (m_pipe / 12) * (
            3 * (r_pvc ** 2 - (r_pvc - 2 * t_pvc) ** 2) + Lpvc_local ** 2
        ) + m_pipe * (L2 + r_pvc) ** 2
        I_local = Iscoop_local + Ipvc_local + Icubesat + Icontainer
        
        # Solve
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
        
        alt_bin = v_ascent * t_bin
        amplitude = (max_val - min_val) / 2
        
        # Store metrics for table (consistent 'Value' key)
        add_metric('L_pvc', f'{lpvc_val:.2f} m', mean, amplitude)
        
        # Plot (metric units)
        label = f'{lpvc_val:.2f} m'
        ax_pvc_mean.plot(alt_bin, mean, color=colors_pvc[i], 
                        linewidth=2.5, label=label, zorder=3-i)
        ax_pvc_amp.plot(alt_bin, amplitude, color=colors_pvc[i], 
                       linewidth=2.5, label=label, zorder=3-i)
    
    # Format PVC plots
    ax_pvc_mean.set_ylabel('Mean Angle (deg)', fontsize=12, fontweight='bold')
    ax_pvc_mean.legend(title='$L_{pvc}$', loc='best', 
                      framealpha=0.95, fontsize=10)
    ax_pvc_mean.set_xlabel('Altitude (m)', fontsize=11)
    
    ax_pvc_amp.set_ylabel('Amplitude (deg)', fontsize=12, fontweight='bold')
    ax_pvc_amp.legend(title='$L_{pvc}$', loc='best', 
                     framealpha=0.95, fontsize=10)
    ax_pvc_amp.set_xlabel('Altitude (m)', fontsize=11)
    
    # -----------------------------
    # ROW 2: Scoop Diameter
    # -----------------------------
    print("Computing scoop diameter effects...")
    for i, d_val in enumerate(scoop_diameters):
        # Setup geometry
        Lpvc_local = Lpvc
        d_scoop_local = d_val
        m_scoop_local = m_scoop
        
        L5 = Lpvc_local / 2
        phi = np.arctan(L2 / L5)
        R3 = L5 / np.cos(phi)
        
        # Update forces
        A_scoop_local = np.pi * (d_scoop_local / 2) ** 2
        F_wind_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * vspan ** 2
        F_ascent_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * v_ascent ** 2
        F_drag_scoop_local = interp1d(xspan, F_wind_scoop, fill_value="extrapolate")
        F_ascent_scoop_local = interp1d(xspan, F_ascent_scoop, fill_value="extrapolate")
        
        # Update inertia
        Iscoop_local = m_scoop_local * (
            (0.4 * ((d_scoop_local / 2) ** 2))
            + (((Lpvc_local / 2 + (d_scoop_local / 2)) ** 2) + (L2 ** 2))
        )
        Ipvc_local = (m_pipe / 12) * (
            3 * (r_pvc ** 2 - (r_pvc - 2 * t_pvc) ** 2) + Lpvc_local ** 2
        ) + m_pipe * (L2 + r_pvc) ** 2
        I_local = Iscoop_local + Ipvc_local + Icubesat + Icontainer
        
        # Solve
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
        
        alt_bin = v_ascent * t_bin
        amplitude = (max_val - min_val) / 2
        
        # Store metrics for table (consistent 'Value' key)
        add_metric('d_scoop', f'{d_val:.4f} m', mean, amplitude)
        
        # Plot (metric units only)
        label = f'{d_val:.4f} m'
        ax_scoop_mean.plot(alt_bin, mean, color=colors_scoop[i], 
                          linewidth=2.5, label=label, zorder=3-i)
        ax_scoop_amp.plot(alt_bin, amplitude, color=colors_scoop[i], 
                         linewidth=2.5, label=label, zorder=3-i)
    
    # Format scoop diameter plots
    ax_scoop_mean.set_ylabel('Mean Angle (deg)', fontsize=12, fontweight='bold')
    ax_scoop_mean.legend(title='$d_{scoop}$', loc='best', 
                        framealpha=0.95, fontsize=10)
    ax_scoop_mean.set_xlabel('Altitude (m)', fontsize=11)
    
    ax_scoop_amp.set_ylabel('Amplitude (deg)', fontsize=12, fontweight='bold')
    ax_scoop_amp.legend(title='$d_{scoop}$', loc='best', 
                       framealpha=0.95, fontsize=10)
    ax_scoop_amp.set_xlabel('Altitude (m)', fontsize=11)
    
    # -----------------------------
    # ROW 3: Scoop Mass
    # -----------------------------
    print("Computing scoop mass effects...")
    for i, m_val in enumerate(scoop_masses):
        # Setup geometry
        Lpvc_local = Lpvc
        d_scoop_local = d_scoop
        m_scoop_local = m_val
        
        L5 = Lpvc_local / 2
        phi = np.arctan(L2 / L5)
        R3 = L5 / np.cos(phi)
        
        # Update forces
        A_scoop_local = np.pi * (d_scoop_local / 2) ** 2
        F_wind_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * vspan ** 2
        F_ascent_scoop = 0.5 * c_dscoop * rhospan * A_scoop_local * v_ascent ** 2
        F_drag_scoop_local = interp1d(xspan, F_wind_scoop, fill_value="extrapolate")
        F_ascent_scoop_local = interp1d(xspan, F_ascent_scoop, fill_value="extrapolate")
        
        # Update inertia
        Iscoop_local = m_scoop_local * (
            (0.4 * ((d_scoop_local / 2) ** 2))
            + (((Lpvc_local / 2 + (d_scoop_local / 2)) ** 2) + (L2 ** 2))
        )
        Ipvc_local = (m_pipe / 12) * (
            3 * (r_pvc ** 2 - (r_pvc - 2 * t_pvc) ** 2) + Lpvc_local ** 2
        ) + m_pipe * (L2 + r_pvc) ** 2
        I_local = Iscoop_local + Ipvc_local + Icubesat + Icontainer
        
        # Solve
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
        
        alt_bin = v_ascent * t_bin
        amplitude = (max_val - min_val) / 2
        
        # Store metrics for table (consistent 'Value' key)
        add_metric('m_scoop', f'{m_val:.2f} kg', mean, amplitude)
        
        # Plot
        label = f'{m_val} kg'
        ax_mass_mean.plot(alt_bin, mean, color=colors_mass[i], 
                         linewidth=2.5, label=label, zorder=3-i)
        ax_mass_amp.plot(alt_bin, amplitude, color=colors_mass[i], 
                        linewidth=2.5, label=label, zorder=3-i)
    
    # Format mass plots
    ax_mass_mean.set_ylabel('Mean Angle (deg)', fontsize=12, fontweight='bold')
    ax_mass_mean.set_xlabel('Altitude (m)', fontsize=11)
    ax_mass_mean.legend(title='$m_{scoop}$', loc='best', 
                       framealpha=0.95, fontsize=10)
    
    ax_mass_amp.set_ylabel('Amplitude (deg)', fontsize=12, fontweight='bold')
    ax_mass_amp.set_xlabel('Altitude (m)', fontsize=11)
    ax_mass_amp.legend(title='$m_{scoop}$', loc='best', 
                      framealpha=0.95, fontsize=10)
    
    plt.savefig('parameter_study_main.png', dpi=300, bbox_inches='tight')
    plt.savefig('parameter_study_main.pdf', bbox_inches='tight')
    print("[OK] Main figure saved as: parameter_study_main.png/.pdf")
    
    # ====================================================================
    # FIGURE 2: Summary Metrics Table
    # ====================================================================
    
    print("\nGenerating summary table...")
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # If there are accidental duplicated column names, drop duplicates (keeps first)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Create figure for table
    fig2, ax_table = plt.subplots(figsize=(13, 6))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create table
    table = ax_table.table(cellText=df.values,
                          colLabels=df.columns,
                          cellLoc='center',
                          loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style the header
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style the rows - color by parameter
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i <= 3:  # Lpvc rows
                cell.set_facecolor('#E8F4F8' if i % 2 == 0 else '#F0F8FF')
            elif i <= 6:  # d_scoop rows
                cell.set_facecolor('#FFE8E8' if i % 2 == 0 else '#FFF0F0')
            else:  # m_scoop rows
                cell.set_facecolor('#FFF4E6' if i % 2 == 0 else '#FFF9F0')
    
    plt.savefig('parameter_study_table.png', dpi=300, bbox_inches='tight')
    print("[OK] Metrics table saved as: parameter_study_table.png")
    
    # Also save as CSV for easy reference
    df.to_csv('parameter_study_metrics.csv', index=False)
    print("[OK] Metrics data saved as: parameter_study_metrics.csv")
    
    print("\n" + "="*60)
    print("SUMMARY METRICS TABLE")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    return fig1, fig2, df


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    print("Starting parameter study analysis")

    fig_main, fig_table, metrics_df = compare_parameter_effects_publication()
    
    print("\nAnalysis complete. Check the output files.")

    plt.show()
