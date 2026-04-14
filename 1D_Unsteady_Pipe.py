"""
1D Unsteady Pipe Flow Analysis (Navier-Stokes)
Comparison of FTCS, BTCS, and Crank-Nicolson Schemes
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Physical and Fluid Constants ---
pressure_diff = -5.12       # Pressure Difference (Pa)
length = 10.0               # Tube Length (m)
grad_press = pressure_diff / length # Pressure Gradient
rho = 1000.0                # Density of fluid (kg/m^3)
viscosity = 0.001           # Dynamic Viscosity (Pa.s)
radius = 0.025              # Radius of the Tube (m)
diameter = 2 * radius       # Tube Diameter (m)
Re = 2000.0                 # Reynolds Number

# --- 2. Calculate Dimensionless Constants ---
u_prime = Re * viscosity / (rho * diameter)
B = (-1) * diameter * grad_press / (rho * pow(u_prime, 2))

t_prime = diameter / u_prime
t_end = 500
M = 10000                   # Number of time steps
time_prime = np.linspace(0, t_end, M)
delta_t = t_end / (M - 1)

# Parameters for Grids
n_radial_list = [5, 25, 50]
results_ftcs = []
results_btcs = []
results_cn = []
radius_arrays = []
radius_prime_arrays = []

print("Starting CFD Solvers (FTCS, BTCS, Crank-Nicolson)...")

# --- 3. Main Loop for Different Grid Sizes ---
for n_rad in n_radial_list:
    
    N = 2 * n_rad - 1
    r_prime_max = 1
    radius_prime = np.linspace(-r_prime_max, r_prime_max, N)
    delta_r = 2 * r_prime_max / (N - 1)
    radius_actual = radius_prime * radius
    
    radius_arrays.append(radius_actual)
    radius_prime_arrays.append(radius_prime)
    
    # ---------------------------------------------------------
    # METHOD 1: EXPLICIT FTCS SCHEME
    # ---------------------------------------------------------
    u_ftcs = np.zeros((N, M))
    for j in range(M - 1):
        for i in range(1, N - 1):
            r_val = radius_prime[i]
            d2u = (u_ftcs[i+1, j] - 2 * u_ftcs[i, j] + u_ftcs[i-1, j]) / (delta_r**2)
            
            # Use L'Hôpital's rule at the center (r=0) to avoid singularity
            if abs(r_val) > 1e-8:
                d1u = (1 / r_val) * (u_ftcs[i+1, j] - u_ftcs[i-1, j]) / (2 * delta_r)
            else:
                d1u = d2u 
                
            term = (4 / Re) * (d2u + d1u)
            u_ftcs[i, j+1] = u_ftcs[i, j] + delta_t * (B + term)
            
    results_ftcs.append(u_ftcs)
    
    # ---------------------------------------------------------
    # METHOD 2: IMPLICIT BTCS SCHEME (TDMA Solver)
    # ---------------------------------------------------------
    u_btcs = np.zeros((N, M))
    a_btcs = np.zeros(N - 2)
    b_btcs = np.zeros(N - 2)
    c_btcs = np.zeros(N - 2)
    
    for i in range(1, N - 1):
        idx = i - 1
        r_val = radius_prime[i]
        if abs(r_val) < 1e-8:
            a_btcs[idx] = -(8 * delta_t) / (Re * delta_r**2)
            b_btcs[idx] = 1 + (16 * delta_t) / (Re * delta_r**2)
            c_btcs[idx] = -(8 * delta_t) / (Re * delta_r**2)
        else:
            a_btcs[idx] = -(4 * delta_t / Re) * (1 / (delta_r**2) - 1 / (2 * r_val * delta_r))
            b_btcs[idx] = 1 + (8 * delta_t) / (Re * delta_r**2)
            c_btcs[idx] = -(4 * delta_t / Re) * (1 / (delta_r**2) + 1 / (2 * r_val * delta_r))
            
    # TDMA Static Forward Elimination
    alfa_btcs = np.zeros(N - 2)
    alfa_btcs[0] = b_btcs[0]
    for i in range(1, N - 2):
        alfa_btcs[i] = b_btcs[i] - (a_btcs[i] * c_btcs[i-1]) / alfa_btcs[i-1]
        
    for j in range(0, M - 1):
        d_btcs = np.zeros(N - 2)
        for i in range(1, N - 1):
            d_btcs[i-1] = u_btcs[i, j] + B * delta_t
            
        gama = np.zeros(N - 2)
        gama[0] = d_btcs[0]
        for i in range(1, N - 2):
            gama[i] = d_btcs[i] - (a_btcs[i] * gama[i-1]) / alfa_btcs[i-1]
            
        u_btcs[N-2, j+1] = gama[N-3] / alfa_btcs[N-3]
        for i in range(N-4, -1, -1):
            u_btcs[i+1, j+1] = (gama[i] - c_btcs[i] * u_btcs[i+2, j+1]) / alfa_btcs[i]
            
    results_btcs.append(u_btcs)
    
    # ---------------------------------------------------------
    # METHOD 3: IMPLICIT CRANK-NICOLSON SCHEME
    # ---------------------------------------------------------
    u_cn = np.zeros((N, M))
    k_cn = 2 * delta_t / Re
    a_cn = np.zeros(N - 2)
    b_cn = np.zeros(N - 2)
    c_cn = np.zeros(N - 2)
    
    for i in range(1, N - 1):
        idx = i - 1
        r_val = radius_prime[i]
        if abs(r_val) < 1e-8:
            a_cn[idx] = -(2 * k_cn) / (delta_r**2)
            b_cn[idx] = 1 + (4 * k_cn) / (delta_r**2)
            c_cn[idx] = -(2 * k_cn) / (delta_r**2)
        else:
            a_cn[idx] = -k_cn * (1 / (delta_r**2) - 1 / (2 * r_val * delta_r))
            b_cn[idx] = 1 + (2 * k_cn) / (delta_r**2)
            c_cn[idx] = -k_cn * (1 / (delta_r**2) + 1 / (2 * r_val * delta_r))
            
    # TDMA Static Forward Elimination
    alfa_cn = np.zeros(N - 2)
    alfa_cn[0] = b_cn[0]
    for i in range(1, N - 2):
        alfa_cn[i] = b_cn[i] - (a_cn[i] * c_cn[i-1]) / alfa_cn[i-1]
        
    for j in range(0, M - 1):
        d_cn = np.zeros(N - 2)
        for i in range(1, N - 1):
            idx = i - 1
            r_val = radius_prime[i]
            if abs(r_val) < 1e-8:
                rhs_a = (2 * k_cn) / (delta_r**2)
                rhs_b = 1 - (4 * k_cn) / (delta_r**2)
                rhs_c = (2 * k_cn) / (delta_r**2)
            else:
                rhs_a = k_cn * (1 / (delta_r**2) - 1 / (2 * r_val * delta_r))
                rhs_b = 1 - (2 * k_cn) / (delta_r**2)
                rhs_c = k_cn * (1 / (delta_r**2) + 1 / (2 * r_val * delta_r))
                
            d_cn[idx] = rhs_a * u_cn[i-1, j] + rhs_b * u_cn[i, j] + rhs_c * u_cn[i+1, j] + B * delta_t
            
        gama = np.zeros(N - 2)
        gama[0] = d_cn[0]
        for i in range(1, N - 2):
            gama[i] = d_cn[i] - (a_cn[i] * gama[i-1]) / alfa_cn[i-1]
            
        u_cn[N-2, j+1] = gama[N-3] / alfa_cn[N-3]
        for i in range(N-4, -1, -1):
            u_cn[i+1, j+1] = (gama[i] - c_cn[i] * u_cn[i+2, j+1]) / alfa_cn[i]
            
    results_cn.append(u_cn)


# --- 4. Plotting Time Evolution (N=50) ---
u_cn_50 = results_cn[-1]
r_prime_50 = radius_prime_arrays[-1]

plt.figure(figsize=(10, 6))
step_size = M // 10
for j in range(0, M, step_size):
    plt.plot(r_prime_50, u_cn_50[:, j], marker='.', label=f'Time (t): {time_prime[j]:.0f}')
plt.plot(r_prime_50, u_cn_50[:, -1], color='black', linewidth=2, label=f'Final State (t={t_end})')

plt.title("Time Evolution of Velocity Profile in Pipe (Crank-Nicolson, N=50)")
plt.xlabel("Dimensionless Radial Position (r/R)")
plt.ylabel("Dimensionless Velocity (u)")
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

# --- 5. Calculating Flow Rates and Shear Stress ---
theo_flow_rate = (abs(pressure_diff) * math.pi * radius**4) / (8 * viscosity * length)
theo_shear_stress = (radius / 2) * (abs(pressure_diff) / length)

print("\n" + "="*50)
print("   RESULTS TABLE (Calculations for N=50)")
print("="*50)

methods = [("FTCS", results_ftcs), ("BTCS", results_btcs), ("Crank-Nicolson", results_cn)]

for name, results in methods:
    u_final_dim = results[-1][:, -1] * u_prime
    r_dim = radius_arrays[-1]
    
    mid_idx = len(r_dim) // 2
    r_half = r_dim[mid_idx:]
    u_half = u_final_dim[mid_idx:]
    
    # Note: using numpy's trapezoid method
    try:
        flow_rate = np.trapezoid(u_half * 2 * math.pi * r_half, r_half)
    except AttributeError:
        # Fallback for older numpy versions
        flow_rate = np.trapz(u_half * 2 * math.pi * r_half, r_half)
    
    delta_r_dim = r_dim[1] - r_dim[0]
    shear_stress = -viscosity * (u_half[-1] - u_half[-2]) / delta_r_dim
    
    print(f"\n--- {name} Method ---")
    print(f"Calculated Flow Rate    : {flow_rate:.8f} m^3/s")
    print(f"Calculated Shear Stress : {shear_stress:.6f} Pa")

print("\n--- Theoretical Values ---")
print(f"Theoretical Flow Rate   : {theo_flow_rate:.8f} m^3/s")
print(f"Theoretical Shear Stress: {theo_shear_stress:.6f} Pa")
print("="*50)
