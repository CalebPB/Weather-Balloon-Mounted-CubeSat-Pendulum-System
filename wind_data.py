# balloon_wind_density_profile_grib.py
# Reads a GRIB file (ERA5 or similar) and outputs wind profile and air density 0–80,000 ft
# Requirements: xarray, cfgrib, numpy, scipy, pandas, matplotlib

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER INPUT ----------------
grib_file = "wind_data.grib"  # Replace with your GRIB filename
lat = 40.0                     # Launch latitude
lon = -105.0                   # Launch longitude
altitudes_ft = np.arange(0, 80001, 2000)  # Altitudes to interpolate (ft)
# --------------------------------------------

# Open GRIB file
ds = xr.open_dataset(grib_file, engine='cfgrib')

# --- Detect vertical coordinate and extract altitude levels ---
if 'heightAboveGround' in ds.coords:
    # HRRR or GFS type (already in meters)
    alt_levels_m = ds['heightAboveGround'].values

elif 'isobaricInhPa' in ds.coords:
    # ERA5 or similar (convert pressure → altitude)
    p = ds['isobaricInhPa'].values  # in hPa
    H0 = 7430.0                     # scale height (m)
    p0 = 1013.25                    # sea-level pressure (hPa)
    alt_levels_m = H0 * np.log(p0 / p)

elif 'level' in ds.coords or 'hybrid' in ds.coords:
    # Model-level data — use level index as placeholder
    alt_levels_m = ds['level'].values

else:
    raise ValueError("No recognizable vertical coordinate found in GRIB file.")

# --- Sort in ascending order (bottom to top) ---
alt_levels_m = np.sort(alt_levels_m)

# --- Optionally convert to feet ---
alt_levels_ft = alt_levels_m * 3.28084

# Now you can use this array as your x_span:
x_span = (alt_levels_m[0], alt_levels_m[-1])  # e.g. for solve_ivp

# Extract pressure levels
if 'isobaricInhPa' in ds.coords:
    p = ds['isobaricInhPa'].values  # pressure in hPa
else:
    raise ValueError("No pressure coordinate found in GRIB file")

# Convert pressure to approximate geopotential height (meters)
H0 = 7430  # scale height in meters
p0 = 1013.25
z_m = H0 * np.log(p0 / p)  # approximate height

# Select nearest point to launch site
u = ds['u'].sel(latitude=lat, longitude=lon, method='nearest').values
v = ds['v'].sel(latitude=lat, longitude=lon, method='nearest').values

# --- Extract temperature for density calculation ---
try:
    # Try common temperature variable names
    if 't' in ds.data_vars:
        T = ds['t'].sel(latitude=lat, longitude=lon, method='nearest').values  # Kelvin
    elif 'temperature' in ds.data_vars:
        T = ds['temperature'].sel(latitude=lat, longitude=lon, method='nearest').values
    elif 'temp' in ds.data_vars:
        T = ds['temp'].sel(latitude=lat, longitude=lon, method='nearest').values
    else:
        raise KeyError("Temperature variable not found")
    
    # Calculate air density using ideal gas law
    R = 287.05  # specific gas constant for dry air (J/(kg·K))
    rho = (p * 100) / (R * T)  # kg/m³
    print("Using temperature data from GRIB file for density calculation")
    
except KeyError:
    print("Temperature not found in GRIB file. Using standard atmosphere model.")
    # Fallback to standard atmosphere
    def standard_atmosphere_density(altitude_m):
        p0 = 101325  # sea level pressure (Pa)
        T0 = 288.15  # sea level temperature (K)
        L = 0.0065   # temperature lapse rate (K/m)
        g = 9.80665  # gravity (m/s²)
        M = 0.0289644  # molar mass of air (kg/mol)
        R_univ = 8.31447    # universal gas constant
        
        T = T0 - L * altitude_m
        p_Pa = p0 * (T / T0) ** (g * M / (R_univ * L))
        rho_val = p_Pa * M / (R_univ * T)
        
        return rho_val
    
    rho = standard_atmosphere_density(z_m)

# Convert altitudes to meters
alts_m = altitudes_ft * 0.3048

# Sort heights for interpolation
order = np.argsort(z_m)
z_sorted = z_m[order]
u_sorted = u[order]
v_sorted = v[order]
rho_sorted = rho[order]

# Interpolate u, v, and rho to requested altitudes
u_interp = interp1d(z_sorted, u_sorted, fill_value="extrapolate")
v_interp = interp1d(z_sorted, v_sorted, fill_value="extrapolate")
rho_interp = interp1d(z_sorted, rho_sorted, fill_value="extrapolate")

u_at_alt = u_interp(alts_m)
v_at_alt = v_interp(alts_m)
rho_at_alt = rho_interp(alts_m)

# Compute wind speed and direction
wind_speed = np.sqrt(u_at_alt**2 + v_at_alt**2)
wind_dir_rad = np.arctan2(-u_at_alt, -v_at_alt)
wind_dir_deg = (np.degrees(wind_dir_rad) + 360) % 360

# Build DataFrame
df = pd.DataFrame({
    "Altitude_ft": altitudes_ft,
    "U_mps": np.round(u_at_alt,2),
    "V_mps": np.round(v_at_alt,2),
    "Wind_speed_mps": np.round(wind_speed,2),
    "Wind_dir_deg_from": np.round(wind_dir_deg,1),
    "Air_density_kg_m3": np.round(rho_at_alt,4)
})

wind_speed_mps = df["Wind_speed_mps"]

# --- INTERPOLATION ---
# Create finer altitude grid (every 10 m)
xspan = np.arange(alts_m.min(), alts_m.max() + 10, 10)  # altitude grid (m)

# Smooth monotonic interpolation (safe for physical data)
wind_interp_func = PchipInterpolator(alts_m, wind_speed_mps)
rho_interp_func = PchipInterpolator(alts_m, rho_at_alt)

# Interpolated wind speed and density at each altitude
vspan = wind_interp_func(xspan)
rhospan = rho_interp_func(xspan)

# --- PRINT RESULTS ---
print("\nxspan (altitude, m):", xspan)
print("\nvspan (wind speed, m/s):", vspan)
print("\nrhospan (air density, kg/m³):", rhospan)
print(f"\nNumber of points: {len(xspan)} (xspan, vspan, and rhospan all have the same length)")

# Optional: Plot both profiles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Wind speed plot
ax1.plot(wind_speed_mps, alts_m, 'o', label='Original data')
ax1.plot(vspan, xspan, '-', label='Interpolated (10 m spacing)')
ax1.set_xlabel('Wind speed (m/s)')
ax1.set_ylabel('Altitude (m)')
ax1.set_title('Wind Speed Profile')
ax1.legend()
ax1.grid(False)

# Air density plot
ax2.plot(rho_at_alt, alts_m, 'o', label='Original data')
ax2.plot(rhospan, xspan, '-', label='Interpolated (10 m spacing)')
ax2.set_xlabel('Air density (kg/m³)')
ax2.set_ylabel('Altitude (m)')
ax2.set_title('Air Density Profile')
ax2.legend()
ax2.grid(False)

plt.tight_layout()
# plt.show()
