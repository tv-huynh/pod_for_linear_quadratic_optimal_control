"""
ES6: main
Original work by: Andrea Petrocchi (July 2023)
Modifications and additions by: Thanh-Van Huynh
"""

import os
import fenics
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized
import supplements, optimization, reduce

# Initialize
p = supplements.analytical_problem()
p.x1a = 0.0; p.x1b = 1.0;   p.x2a = 0.0; p.x2b = 1.0
p.t0 = 0;                   p.T = 2
p.h = 0.05;                 p.K = 101
p.y0 = fenics.Constant(0.0)
p.c_u = fenics.Constant(1.0)
y_d_exp = fenics.Constant(1.0)
u_d_exp = fenics.Constant(0.0)

m = supplements.parabolic_model(p)
m.build_problem()
opt = optimization.optimization_class(m)
opt.beta = 1.e-3
opt.tol = 1.e-7
y_d_0 = fenics.interpolate( y_d_exp, m.V ).vector()[:]
opt.Y_d = np.repeat( y_d_0.reshape(-1,1), m.K, axis=1 )
u_d_0 = fenics.interpolate( u_d_exp, m.V ).vector()[:]
opt.U_d = np.repeat( u_d_0.reshape(-1,1), m.K, axis=1 )
U_0 = opt.U_d.copy()

GENERATE_PLOTS = True
if GENERATE_PLOTS:
    PLOTS = "plots/"; m.format_folder(PLOTS)
    
#============================================================
#%% FOM
#============================================================
# Solve FOM
print("="*60)
print("SOLVING FULL-ORDER MODEL (FOM)")
print("="*60)

u_opt_BB, history = opt.solve( U_0, "BB",
                        print_info=True,
                        print_final=True,
                        plot_grad_convergence=True,
                        save_plot_grad_convergence=GENERATE_PLOTS,
                        path=PLOTS+"convergence_BB_FOM",
                    )
U_opt_BB = m.vector_to_matrix(u_opt_BB,option="control")
Y_opt_BB = m.solve_state(U_opt_BB)
P_opt_BB = m.solve_adjoint(opt.Y_d-Y_opt_BB)

print(f"\nFOM Results:")
print(f"  State dimension: {Y_opt_BB.shape[0]}")
print(f"  Control dimension: {U_opt_BB.shape[0]}")
print(f"  Time steps: {Y_opt_BB.shape[1]}")

#============================================================
#%% ROM
#============================================================
#%% Construct ROM out of FOM snapshots
print("\n" + "="*60)
print("CONSTRUCTING REDUCED-ORDER MODEL (ROM)")
print("="*60)

l = 10
optimal_snapshots = True

# Get snapshots
if optimal_snapshots == True: # train with optimal snapshots
    snapshots = [Y_opt_BB, P_opt_BB]
    print("Using optimal FOM snapshots for POD basis")
else: # train with initial snapshots
    snapshots = [m.get_snapshots(U_0,opt.Y_d)]
    print("Using initial snapshots for POD basis")
print("Doing POD with l="+str(l)+" POD basis vectors")

# Do POD
pod = reduce.pod(m)
POD_Basis, POD_values = pod.pod_basis(snapshots,l)
pod.project(POD_Basis)
opt_ROM = optimization.optimization_class(pod.model)
opt_ROM.beta = opt.beta
opt_ROM.tol = opt.tol
opt_ROM.Y_d = POD_Basis.T @ opt.Y_d # project Y_d into reduced space
opt_ROM.U_d = POD_Basis.T @ opt.U_d #project U_d into reduced space
U_0_ROM = POD_Basis.T @ U_0 # project U_0 into reduced space

print(f"  Reduced Y_d shape: {opt_ROM.Y_d.shape}")
print(f"  Reduced U_d shape: {opt_ROM.U_d.shape}")
print(f"Initial guess U_0_ROM shape: {U_0_ROM.shape}")

# Check energy captured
total_energy = np.sum(POD_values)
energy_l = np.sum(POD_values[:len(POD_values)])
print(f"\nPOD basis captures {energy_l/total_energy*100:.2f}% of total energy")

# Solve ROM 
print("\n" + "="*60)
print("SOLVING REDUCED-ORDER MODEL (ROM)")
print("="*60)

u_BB_ROM, history_BB_ROM = opt_ROM.solve( U_0_ROM, "BB",
                            print_info=True,
                            print_final=True,
                            plot_grad_convergence=True,
                            save_plot_grad_convergence=GENERATE_PLOTS,
                            path=PLOTS+"convergence_BB_ROM",
                    )
U_BB_ROM = pod.model.vector_to_matrix(u_BB_ROM,option="control")
Y_BB_ROM = pod.model.solve_state(U_BB_ROM)
P_BB_ROM = pod.model.solve_adjoint(opt_ROM.Y_d - Y_BB_ROM)

print(f"\nROM Results (before recovery):")
print(f"  Reduced state dimension: {Y_BB_ROM.shape[0]}")
print(f"  Full control dimension: {U_BB_ROM.shape[0]}")
print(f"  Time steps: {Y_BB_ROM.shape[1]}")

# Recover FOM solution from ROM
U_BB_ROM_full = POD_Basis @ U_BB_ROM  # Project control back to full space
Y_BB_ROM_full = POD_Basis @ Y_BB_ROM  # Project state back to full space
P_BB_ROM_full = POD_Basis @ P_BB_ROM  # Project adjoint back to full space

print(f"\nROM Results (after recovery to full space):")
print(f"  State dimension: {Y_BB_ROM_full.shape[0]}")
print(f"  Control dimension: {U_BB_ROM_full.shape[0]}")
print(f"  Adjoint dimension: {P_BB_ROM_full.shape[0]}")

#============================================================
#%% Error analysis
#============================================================
#%% Compute errors
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Compute errors using FOM matrices
Y_diff = Y_opt_BB - Y_BB_ROM_full
U_diff = U_opt_BB - U_BB_ROM_full
P_diff = P_opt_BB - P_BB_ROM_full

m.M = m.M_FOM
m.A = m.A_FOM
m.update_state_products()

state_error = m.eval_L2H_norm(Y_diff, space_norm="L2")
state_norm = m.eval_L2H_norm(Y_opt_BB, space_norm="L2")
state_rel_error = state_error / state_norm

control_error = m.eval_L2H_norm(U_diff, space_norm="L2")
control_norm = m.eval_L2H_norm(U_opt_BB, space_norm="L2")
control_rel_error = control_error / control_norm

adjoint_error = m.eval_L2H_norm(P_diff, space_norm="L2")
adjoint_norm = m.eval_L2H_norm(P_opt_BB, space_norm="L2")
adjoint_rel_error = adjoint_error / adjoint_norm

print(f"Relative state error:   {state_rel_error:.6e}")
print(f"Relative control error: {control_rel_error:.6e}")
print(f"Relative adjoint error: {adjoint_rel_error:.6e}")

print(f"\nFOM optimization time: {history['time']:.3f} seconds")
print(f"ROM optimization time: {history_BB_ROM['time']:.3f} seconds")
print(f"Speedup factor: {history['time']/history_BB_ROM['time']:.2f}x")

#============================================================
#%% Plots
#============================================================
if GENERATE_PLOTS:
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Plot POD eigenvalue decay
    pod.plot_pod_values(path=PLOTS)
    
    for k in range(0, p.K, 10):
        print(f"Plotting time step {k}...")
        m.plot_3d(Y_opt_BB[:,k], title=f"FOM State t={k}", save_png=True, path=PLOTS+f"Y_FOM_{k}.png")
        m.plot_3d(U_opt_BB[:,k], title=f"FOM Control t={k}", save_png=True, path=PLOTS+f"U_FOM_{k}.png")
        m.plot_3d(P_opt_BB[:,k], title=f"FOM Adjoint t={k}", save_png=True, path=PLOTS+f"P_FOM_{k}.png")
        m.plot_3d(Y_BB_ROM_full[:,k], title=f"ROM State t={k}", save_png=True, path=PLOTS+f"Y_ROM_{k}.png")
        m.plot_3d(U_BB_ROM_full[:,k], title=f"ROM Control t={k}", save_png=True, path=PLOTS+f"U_ROM_{k}.png")
        m.plot_3d(P_BB_ROM_full[:,k], title=f"ROM Adjoint t={k}", save_png=True, path=PLOTS+f"P_ROM_{k}.png")
    
    # Final time step
    m.plot_3d(Y_opt_BB[:,p.K-2], title=f"FOM State t={p.K-2}", save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}.png")
    m.plot_3d(U_opt_BB[:,p.K-2], title=f"FOM Control t={p.K-2}", save_png=True, path=PLOTS+f"U_FOM_{p.K-2}.png")
    m.plot_3d(P_opt_BB[:,p.K-2], title=f"FOM Adjoint t={p.K-2}", save_png=True, path=PLOTS+f"P_FOM_{p.K-2}.png")
    m.plot_3d(Y_BB_ROM_full[:,p.K-2], title=f"ROM State t={p.K-2}", save_png=True, path=PLOTS+f"Y_ROM_{p.K-2}.png")
    m.plot_3d(U_BB_ROM_full[:,p.K-2], title=f"ROM Control t={p.K-2}", save_png=True, path=PLOTS+f"U_ROM_{p.K-2}.png")
    m.plot_3d(P_BB_ROM_full[:,p.K-2], title=f"ROM Adjoint t={p.K-2}", save_png=True, path=PLOTS+f"P_ROM_{p.K-2}.png")
    
    print("All plots saved to " + PLOTS)

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)