"""
main.py
Original work by: Andrea Petrocchi (July 2023) -> Initializing, solving FOM
Modifications and additions by: Thanh-Van Huynh -> Varying betas, solving ROM, doing error analysis
"""

import warnings
import fenics
import numpy as np
from scipy.sparse import SparseEfficiencyWarning
import supplements, optimization, reduce

# Suppress SparseEfficiencyWarning
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)

# Initialize
VARY_BETA = False
DO_ERROR_ANALYSIS = True
GENERATE_PLOTS = True

beta = 1.e-3 # regularization factor in the cost functional
tol = 1.e-7 # tolerance for the optimization algorithm

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
opt = optimization.optimization_class(m,beta,tol)
y_d_0 = fenics.interpolate( y_d_exp, m.V ).vector()[:]
opt.Y_d = np.repeat( y_d_0.reshape(-1,1), m.K, axis=1 )
u_d_0 = fenics.interpolate( u_d_exp, m.V ).vector()[:]
opt.U_d = np.repeat( u_d_0.reshape(-1,1), m.K, axis=1 )
U_0 = opt.U_d.copy()

if GENERATE_PLOTS:
    PLOTS = "plots/"
    m.format_folder(PLOTS)
    
#============================================================
#%% FOM
#============================================================
# Solve FOM
print("="*60)
print("SOLVING FULL-ORDER MODEL (FOM)")
print("="*60)

u_opt, history = opt.solve( U_0, "BB",
                        print_info=True,
                        print_final=True,
                        plot_grad_convergence=True,
                        save_plot_grad_convergence=GENERATE_PLOTS,
                        path=PLOTS+"convergence_FOM",
                    )
U_opt = m.vector_to_matrix(u_opt,option="control")
Y_opt = m.solve_state(U_opt)
P_opt = m.solve_adjoint(opt.Y_d-Y_opt)

print(f"\nFOM Results:")
print(f"  State dimension: {Y_opt.shape[0]}")
print(f"  Control dimension: {U_opt.shape[0]}")
print(f"  Time steps: {Y_opt.shape[1]}")

#============================================================
#%% FOM: different beta
#============================================================
if VARY_BETA:
    print("\n" + "="*60)
    print("SOLVING FULL-ORDER MODEL (FOM) FOR DIFFERENT BETA")
    print("="*60)

    number_of_betas = 5
    beta_list = [10.**(-j) for j in range(0,number_of_betas)]

    for j in range(0,number_of_betas):
        print("\n" + "-"*60)
        print("beta = "+str(j))
        print("-"*60)
        beta = beta_list[j]
        opt_beta = optimization.optimization_class(m,beta,tol)
        opt_beta.Y_d = np.repeat( y_d_0.reshape(-1,1), m.K, axis=1 )
        opt_beta.U_d = np.repeat( u_d_0.reshape(-1,1), m.K, axis=1 )

        u_opt_beta, history_beta = opt_beta.solve( U_0, "BB",
                            print_info=True,
                            print_final=True,
                            plot_grad_convergence=True,
                            save_plot_grad_convergence=GENERATE_PLOTS,
                            path=PLOTS+"convergence_beta"+str(j),
                        )
        U_opt_beta = m.vector_to_matrix(u_opt_beta,option="control")
        Y_opt_beta = m.solve_state(U_opt_beta)
        P_opt_beta = m.solve_adjoint(opt_beta.Y_d-Y_opt_beta)

        if GENERATE_PLOTS: # final time step
            m.plot_3d(U_opt_beta[:,p.K-2], save_png=True, path=PLOTS+f"U_FOM_{p.K-2}_beta{j}.png") # title=f"FOM Control t={p.K-2} for $\beta$={beta:.0e}"
            m.plot_3d(Y_opt_beta[:,p.K-2], save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}_beta{j}.png") # title=f"FOM State t={p.K-2} for $\beta$={beta:.0e}"
            m.plot_3d(P_opt_beta[:,p.K-2], save_png=True, path=PLOTS+f"P_FOM_{p.K-2}_beta{j}.png") # title=f"FOM Adjoint t={p.K-2} for $\beta$={beta:.0e}"
    
    beta = 1.e-3

#============================================================
#%% ROM
#============================================================
#%% Construct ROM out of FOM snapshots
print("\n" + "="*60)
print("CONSTRUCTING REDUCED-ORDER MODEL (ROM)")
print("="*60)

l = 20
optimal_snapshots = True
if optimal_snapshots == True:
    file_name = "optimalSnapshots"
else:
    file_name = "initalSnapshots"

# Get snapshots
if optimal_snapshots == True: # train with optimal snapshots
    snapshots = [Y_opt, P_opt]
    print("Using optimal FOM snapshots for POD basis")
else: # train with initial snapshots
    snapshots = [m.get_snapshots(U_0,opt.Y_d)]
    print("Using initial snapshots for POD basis")
print("Doing POD with l="+str(l)+" POD basis vectors")

# Do POD
pod = reduce.pod(m)
POD_Basis, POD_values = pod.pod_basis(snapshots,l)
Y_d_proj, U_d_proj, U_0_ROM = pod.project(POD_Basis,opt.Y_d,opt.U_d,U_0)
opt_ROM = optimization.optimization_class(pod.model,beta,tol)
opt_ROM.Y_d = Y_d_proj
opt_ROM.U_d = U_d_proj

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

u_ROM, history_ROM = opt_ROM.solve( U_0_ROM, "BB",
                            print_info=True,
                            print_final=True,
                            plot_grad_convergence=True,
                            save_plot_grad_convergence=GENERATE_PLOTS,
                            path=PLOTS+"convergence_ROM",
                    )
U_ROM = pod.model.vector_to_matrix(u_ROM,option="control")
Y_ROM = pod.model.solve_state(U_ROM)
P_ROM = pod.model.solve_adjoint(opt_ROM.Y_d - Y_ROM)

# print(f"\nROM Results (before recovery):")
# print(f"  Reduced state dimension: {Y_ROM.shape[0]}")
# print(f"  Full control dimension: {U_ROM.shape[0]}")
# print(f"  Time steps: {Y_ROM.shape[1]}")

# Recover FOM solution from ROM
U_ROM_full = POD_Basis @ U_ROM  # Project control back to full space
Y_ROM_full = POD_Basis @ Y_ROM  # Project state back to full space
P_ROM_full = POD_Basis @ P_ROM  # Project adjoint back to full space

# print(f"\nROM Results (after recovery to full space):")
# print(f"  State dimension: {Y_ROM_full.shape[0]}")
# print(f"  Control dimension: {U_ROM_full.shape[0]}")
# print(f"  Adjoint dimension: {P_ROM_full.shape[0]}")

print(f"\nFOM optimization time: {history['time']:.3f} seconds")
print(f"ROM optimization time: {history_ROM['time']:.3f} seconds")
print(f"Speedup factor: {history['time']/history_ROM['time']:.2f}x")
pod.plot_pod_values(path=PLOTS)

#============================================================
#%% Error analysis
#============================================================
#%% Compute errors
if DO_ERROR_ANALYSIS:
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    space_norm = "L2"
    control_error_list = []

    for j in range(1,pod.basissize+1):
        print("\n" + "-"*60)
        print("Number of snapshots l="+str(j))
        print("-"*60)
        m_err = supplements.parabolic_model(p)
        m_err.build_problem()
        pod_err = reduce.pod(m_err)

        POD_basis_err, POD_values_err = pod_err.pod_basis(snapshots,j)
        Y_d_err, U_d_err, U_0_ROM_err= pod_err.project(POD_basis_err,opt.Y_d,opt.U_d,U_0)
        opt_ROM_err = optimization.optimization_class(pod_err.model,beta,tol)
        opt_ROM_err.Y_d = Y_d_err
        opt_ROM_err.U_d = U_d_err

        # print(f"  Reduced Y_d shape: {opt_ROM_err.Y_d.shape}")
        # print(f"  Reduced U_d shape: {opt_ROM_err.U_d.shape}")
        # print(f"Initial guess U_0_ROM shape: {U_0_ROM_err.shape}")

        # Solve ROM 
        print("\nSOLVING REDUCED-ORDER MODEL (ROM)")

        u_BB_ROM_err, _ = opt_ROM_err.solve( U_0_ROM_err, "BB",
                                    print_info=True,
                                    print_final=True,
                                    plot_grad_convergence=True,
                                    save_plot_grad_convergence=GENERATE_PLOTS,
                                    path=PLOTS+"convergence_ROM_"+str(j)+"_snapshots",
                            )
        U_BB_ROM_err = pod_err.model.vector_to_matrix(u_BB_ROM_err,option="control")

        # Recover FOM solution from ROM
        U_BB_ROM_full_err = POD_basis_err @ U_BB_ROM_err  # Project control back to full space

        # Compute error using FOM matrices
        U_diff = U_BB_ROM_full_err - U_opt

        m_err.M = m_err.M_FOM
        m_err.A = m_err.A_FOM
        m_err.update_state_products()

        control_error = m_err.eval_L2H_norm(U_diff, space_norm)
        control_error_list.append(control_error)

        print(f"Control error: {control_error:.6e}")

    if GENERATE_PLOTS:
        pod.plot_error(control_error_list,path=PLOTS)

#============================================================
#%% Plots
#============================================================
if GENERATE_PLOTS:
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    for k in range(0, p.K, 10):
        print(f"Plotting time step {k}...")
        m.plot_3d(Y_opt[:,k], save_png=True, path=PLOTS+f"Y_FOM_{k}.png") # title=f"FOM State t={k}"
        m.plot_3d(U_opt[:,k], save_png=True, path=PLOTS+f"U_FOM_{k}.png") # title=f"FOM Control t={k}"
        m.plot_3d(P_opt[:,k], save_png=True, path=PLOTS+f"P_FOM_{k}.png") # title=f"FOM Adjoint t={k}"
        m.plot_3d(Y_ROM_full[:,k], save_png=True, path=PLOTS+f"Y_ROM_{k}.png") # title=f"ROM State t={k}"
        m.plot_3d(U_ROM_full[:,k], save_png=True, path=PLOTS+f"U_ROM_{k}.png") # title=f"ROM Control t={k}"
        m.plot_3d(P_ROM_full[:,k], save_png=True, path=PLOTS+f"P_ROM_{k}.png") # title=f"ROM Adjoint t={k}"
    
    # Final time step
    m.plot_3d(Y_opt[:,p.K-2], save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}.png") # title=f"FOM State t={p.K-2}"
    m.plot_3d(U_opt[:,p.K-2], save_png=True, path=PLOTS+f"U_FOM_{p.K-2}.png") # title=f"FOM Control t={p.K-2}"
    m.plot_3d(P_opt[:,p.K-2], save_png=True, path=PLOTS+f"P_FOM_{p.K-2}.png") # title=f"FOM Adjoint t={p.K-2}"
    m.plot_3d(Y_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"Y_ROM_{p.K-2}.png") # title=f"ROM State t={p.K-2}"
    m.plot_3d(U_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"U_ROM_{p.K-2}.png") # title=f"ROM Control t={p.K-2}"
    m.plot_3d(P_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"P_ROM_{p.K-2}.png") # title=f"ROM Adjoint t={p.K-2}"
    
    print("All plots saved to " + PLOTS)

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)