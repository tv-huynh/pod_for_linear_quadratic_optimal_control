"""
main.py
@author: Thanh-Van Huynh
"""

import warnings
import fenics
import numpy as np
from scipy.sparse import SparseEfficiencyWarning
import supplements, optimization, reduce

# Suppress SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# Initialize
SOLVE_FOM = True
SOLVE_ROM = False
DIFFERENT_SIGMAS = False
DIFFERENT_CONTROL_DOMAINS = False
REDUCED_SPACE_VARIANTS = False
DIFFERENT_PDE_PARAMETERS = False
GENERATE_PLOTS = True

space_norm = "L2" # L2, H1, H10
l = 10
optimal_snapshots = True

sigma = 1.e-1; sigma_og = sigma.copy() # regularization factor in the cost functional
tol_abs = 1.e-4; tol_rel = 1.e-3 # tolerance for the optimization algorithm
print("\n" + "="*60)
print("SIMULATION STARTING")
print("="*60)
print("Regularization: "+str(sigma))

#============================================================
#%% Build Model
#============================================================

p = supplements.analytical_problem()
p.x1a = 0.0; p.x1b = 1.0;   p.x2a = 0.0; p.x2b = 1.0
#p.omega = [((0.25, 0.75), (0.25, 0.75))] # central
#p.omega = [((0.0, 0.5), (0.0, 1.0))] # upwind
#p.omega = [((0.5, 1.0), (0.0, 1.0))] # downwind
#p.omega = [((0.0, 0.5), (0.0, 0.5)), ((0.5,1.0),(0.5,1.0))] # disconnected
p.t0 = 0;                   p.T = 2
p.h = 0.05;                 p.K = 101
p.y0 = fenics.Constant(0.0)
p.kappa = fenics.Constant(1.0)
p.beta_x1 = fenics.Constant(1.0)
p.beta_x2 = fenics.Constant(0.0)
#p.beta_x1 = fenics.Expression("-pi*sin(pi*x[0])*cos(pi*x[1])",degree=2) # one swirl
#p.beta_x2 = fenics.Expression("pi*cos(pi*x[0])*sin(pi*x[1])",degree=2)
p.beta = fenics.as_vector((p.beta_x1, p.beta_x2))
p.gamma = fenics.Constant(1.0)
#p.f = fenics.Expression("(1.0 + 8.0*pow(pi,2)*t)*sin(2*pi*x[0])*sin(2*pi*x[1])",t=0.0,degree=3)
p.f = 0
y_d = fenics.Constant(1.0)
u_d = fenics.Constant(0.0)
u_0 = fenics.Constant(1.0)

m = supplements.parabolic_model(p)
m.build_problem()
opt = optimization.optimization_class(m,sigma,tol_abs,tol_rel)
y_d = fenics.interpolate( y_d, m.V ).vector()[:]; opt.Y_d = np.repeat( y_d.reshape(-1,1), m.K, axis=1 )
u_d = fenics.interpolate( u_d, m.V ).vector()[:]; opt.U_d = np.repeat( u_d.reshape(-1,1), m.K, axis=1 )
u_0 = fenics.interpolate( u_0, m.V ).vector()[:]; U_0 = np.repeat( u_0.reshape(-1,1), m.K, axis=1 )

if GENERATE_PLOTS:
    PLOTS = "plots/"
    m.format_folder(PLOTS)
    m.plot_beta(title=None, save_png=True, path=PLOTS+"beta_field.png")
    m.plot_mesh(path=PLOTS+"mesh.png")
    # m.plot_f_at_index(50, save_png=True, path=PLOTS+f"f_50.png") # forcing term at k=50, i.e., t=1

#============================================================
#%% FOM
#============================================================
if SOLVE_FOM:
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

    if GENERATE_PLOTS:
        #m.plot_contour_clean(P_opt[:,50], path=PLOTS) 
        # First inner time step
        """
        m.plot_3d(Y_opt[:,1], save_png=True, path=PLOTS+"Y_FOM_1.png") 
        m.plot_3d(U_opt[:,1], save_png=True, path=PLOTS+"U_FOM_1.png") 
        m.plot_3d(P_opt[:,1], save_png=True, path=PLOTS+"P_FOM_1.png") 
        # Middle time step
        m.plot_3d(Y_opt[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_FOM_{int((p.K-1)/2)}.png") 
        m.plot_3d(U_opt[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_FOM_{int((p.K-1)/2)}.png") 
        m.plot_3d(P_opt[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_FOM_{int((p.K-1)/2)}.png") 
        # Final inner time step
        m.plot_3d(Y_opt[:,p.K-2], save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}.png") 
        m.plot_3d(U_opt[:,p.K-2], save_png=True, path=PLOTS+f"U_FOM_{p.K-2}.png") 
        m.plot_3d(P_opt[:,p.K-2], save_png=True, path=PLOTS+f"P_FOM_{p.K-2}.png") """

#============================================================
#%% ROM
#============================================================
if SOLVE_ROM:
    # Construct ROM out of FOM snapshots
    print("\n" + "="*60)
    print("CONSTRUCTING REDUCED-ORDER MODEL (ROM)")
    print("="*60)

    # Get snapshots
    if optimal_snapshots == True: # train with optimal snapshots
        snapshots = [Y_opt, P_opt]
        print("Using optimal FOM snapshots for POD basis")
    else: # train with initial snapshots
        snapshots = m.get_snapshots(U_0,opt.Y_d)
        print("Using initial snapshots for POD basis")
    print("Doing POD with l="+str(l)+" POD basis vectors")

    # Do POD
    rom = reduce.pod(m,space_norm)
    POD_Basis, POD_values = rom.pod_basis(snapshots,l)
    Y_d_proj, U_d_proj, U_0_ROM = rom.project(POD_Basis,opt.Y_d,opt.U_d,U_0)
    opt_ROM = optimization.optimization_class(rom.model,sigma,tol_abs,tol_rel)
    opt_ROM.Y_d = Y_d_proj
    opt_ROM.U_d = U_d_proj

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
    U_ROM = rom.model.vector_to_matrix(u_ROM,option="control")
    Y_ROM = rom.model.solve_state(U_ROM)
    P_ROM = rom.model.solve_adjoint(opt_ROM.Y_d - Y_ROM)

    print(f"\nROM Results:")
    print(f"  Reduced state dimension: {Y_ROM.shape[0]}")
    print(f"  Reduced control dimension: {U_ROM.shape[0]}")

    # Recover FOM solution from ROM
    U_ROM_full = POD_Basis @ U_ROM  # Project control back to full space
    Y_ROM_full = POD_Basis @ Y_ROM  # Project state back to full space
    P_ROM_full = POD_Basis @ P_ROM  # Project adjoint back to full space

    rom_time = history_ROM['time'] + rom.offline_time_cholesky + rom.offline_time_basisconstruction + rom.offline_time_projection
    print(f"\nFOM optimization time: {history['time']:.3f} seconds")
    print(f"ROM optimization time (offline+solving): {rom_time:.3f} seconds")
    print(f"Speedup factor: {history['time']/rom_time:.2f}x")
    
    if GENERATE_PLOTS:
        rom.plot_pod_values(path=PLOTS)

#============================================================
#%% Do error analysis for different sigmas
#============================================================
if DIFFERENT_SIGMAS:
    print("\n" + "="*60)
    print("ERROR ANALYSIS FOR DIFFERENT SIGMA")
    print("="*60)

    sigma_list = [1.e-3, 1.e-2, 1.e-1]

    control_errors = []; rel_control_errors = []
    state_errors = []; rel_state_errors = []
    adjoint_errors = []; rel_adjoint_errors = []

    fom = supplements.parabolic_model(p)
    fom.build_problem()

    for sigma in sigma_list:
        print("\n" + "-"*60)
        print("sigma = "+str(sigma))
        print("-"*60)

        exp = int(np.round(np.log10(sigma)))

        # Initialize FOM
        opt_FOM = optimization.optimization_class(fom,sigma,tol_abs,tol_rel)
        opt_FOM.Y_d = np.repeat( y_d.reshape(-1,1), fom.K, axis=1 )
        opt_FOM.U_d = np.repeat( u_d.reshape(-1,1), fom.K, axis=1 )

        # Solve FOM
        print("\nSOLVING FULL-ORDER MODEL (FOM)")
        u_opt_FOM, history_FOM = opt_FOM.solve( U_0, "BB",
                            print_info=True,
                            print_final=True,
                            plot_grad_convergence=True,
                            save_plot_grad_convergence=GENERATE_PLOTS,
                            path=PLOTS+"convergence_FOM_sigma"+str(exp),
                        )
        U_opt_FOM = fom.vector_to_matrix(u_opt_FOM,option="control")
        Y_opt_FOM = fom.solve_state(U_opt_FOM)
        P_opt_FOM = fom.solve_adjoint(opt_FOM.Y_d-Y_opt_FOM)

        # Get snapshots
        if optimal_snapshots == True: # train with optimal snapshots
            snapshots = [Y_opt_FOM, P_opt_FOM]
            print("\nUsing optimal FOM snapshots for POD basis")
        else: # train with initial snapshots

            snapshots = fom.get_snapshots(U_0,opt_FOM.Y_d)
            print("\nUsing initial snapshots for POD basis")

        # Initialize ROM
        rom = supplements.parabolic_model(p)
        rom.build_problem()
        pod = reduce.pod(rom)
        POD_basis, POD_values = pod.pod_basis(snapshots,l)
        Y_d_ROM, U_d_ROM, U_0_ROM= pod.project(POD_basis,opt_FOM.Y_d,opt_FOM.U_d,U_0)
        opt_ROM = optimization.optimization_class(pod.model,sigma,tol_abs,tol_rel)
        opt_ROM.Y_d = Y_d_ROM
        opt_ROM.U_d = U_d_ROM

        # Solve ROM 
        print("\nSOLVING REDUCED-ORDER MODEL (ROM)")
        u_ROM, history_ROM = opt_ROM.solve( U_0_ROM, "BB",
                                    print_info=True,
                                    print_final=True,
                                    plot_grad_convergence=True,
                                    save_plot_grad_convergence=GENERATE_PLOTS,
                                    path=PLOTS+"convergence_ROM_sigma"+str(exp),
                            )
        U_opt_ROM = pod.model.vector_to_matrix(u_ROM,option="control")
        Y_opt_ROM = pod.model.solve_state(U_opt_ROM)
        P_opt_ROM = pod.model.solve_adjoint(opt_ROM.Y_d - Y_opt_ROM)

        # Project ROM solutions into full space
        U_ROM_full = POD_basis @ U_opt_ROM  
        Y_ROM_full = POD_basis @ Y_opt_ROM
        P_ROM_full = POD_basis @ P_opt_ROM

        # Compute (absolute and relative) errors in full space
        control_error = fom.eval_L2H_norm(U_ROM_full - U_opt_FOM, space_norm="control")
        state_error = fom.eval_L2H_norm(Y_ROM_full - Y_opt_FOM, space_norm)
        adjoint_error = fom.eval_L2H_norm(P_ROM_full - P_opt_FOM, space_norm)
        rel_control_error = fom.eval_L2H_norm(U_ROM_full - U_opt_FOM, space_norm="control") / m.eval_L2H_norm(U_opt_FOM, space_norm="control")
        rel_state_error = fom.eval_L2H_norm(Y_ROM_full - Y_opt_FOM, space_norm) / m.eval_L2H_norm(Y_opt_FOM, space_norm)
        rel_adjoint_error = fom.eval_L2H_norm(P_ROM_full - P_opt_FOM, space_norm) / m.eval_L2H_norm(P_opt_FOM, space_norm)

        control_errors.append(control_error); rel_control_errors.append(rel_control_error)
        state_errors.append(state_error); rel_state_errors.append(rel_state_error)
        adjoint_errors.append(adjoint_error); rel_adjoint_errors.append(rel_adjoint_error)

        # Print results to console
        print(f"\nControl error: {control_error:.6e}")
        print(f"State error: {state_error:.6e}")
        print(f"Adjoint error: {adjoint_error:.6e}")

        print(f"\nRelative control error: {rel_control_error:.6e}")
        print(f"Relative state error: {rel_state_error:.6e}")
        print(f"Relative adjoint error: {rel_adjoint_error:.6e}")

        rom_time = history_ROM['time'] + pod.offline_time_cholesky + pod.offline_time_basisconstruction + pod.offline_time_projection
    
        print("\nROM times:")
        print("ROM cholesky: "+str(pod.offline_time_cholesky)+" seconds")
        print("ROM basis construction: "+str(pod.offline_time_basisconstruction)+" seconds")
        print("ROM projection: "+str(pod.offline_time_projection)+" seconds")
        print("ROM solving: "+str(history_ROM['time'])+" seconds")

        print(f"\nFOM optimization time: {history_FOM['time']:.2f} seconds")
        print(f"ROM optimization time: {rom_time:.2f} seconds")
        print(f"Speedup factor: {history_FOM['time']/rom_time:.2f}x")

        if GENERATE_PLOTS:
            # first inner time step
            fom.plot_3d(U_opt_FOM[:,1], save_png=True, path=PLOTS+f"U_FOM_1_sigma{exp}.png")
            fom.plot_3d(Y_opt_FOM[:,1], save_png=True, path=PLOTS+f"Y_FOM_1_sigma{exp}.png")
            fom.plot_3d(P_opt_FOM[:,1], save_png=True, path=PLOTS+f"P_FOM_1_sigma{exp}.png")
            fom.plot_3d(U_ROM_full[:,1], save_png=True, path=PLOTS+f"U_ROM_1_sigma{exp}.png")
            fom.plot_3d(Y_ROM_full[:,1], save_png=True, path=PLOTS+f"Y_ROM_1_sigma{exp}.png")
            fom.plot_3d(P_ROM_full[:,1], save_png=True, path=PLOTS+f"P_ROM_1_sigma{exp}.png")
            # middle time step
            fom.plot_3d(U_opt_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_FOM_{int((p.K-1)/2)}_sigma{exp}.png")
            fom.plot_3d(Y_opt_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_FOM_{int((p.K-1)/2)}_sigma{exp}.png")
            fom.plot_3d(P_opt_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_FOM_{int((p.K-1)/2)}_sigma{exp}.png")
            fom.plot_3d(U_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_ROM_{int((p.K-1)/2)}_sigma{exp}.png")
            fom.plot_3d(Y_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_ROM_{int((p.K-1)/2)}_sigma{exp}.png")
            fom.plot_3d(P_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_ROM_{int((p.K-1)/2)}_sigma{exp}.png")
            # final inner time step
            fom.plot_3d(U_opt_FOM[:,p.K-2], save_png=True, path=PLOTS+f"U_FOM_{p.K-2}_sigma{exp}.png")
            fom.plot_3d(Y_opt_FOM[:,p.K-2], save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}_sigma{exp}.png")
            fom.plot_3d(P_opt_FOM[:,p.K-2], save_png=True, path=PLOTS+f"P_FOM_{p.K-2}_sigma{exp}.png")
            fom.plot_3d(U_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"U_ROM_{p.K-2}_sigma{exp}.png")
            fom.plot_3d(Y_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"Y_ROM_{p.K-2}_sigma{exp}.png")
            fom.plot_3d(P_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"P_ROM_{p.K-2}_sigma{exp}.png")

    if GENERATE_PLOTS:
        print("\nGenerating error plots...")
        m.plot_error_vs_x(sigma_list,control_errors,state_errors,adjoint_errors,axis="loglog",x_axis=r'$\sigma$',path=PLOTS+"err_vs_sigma_abs")
        m.plot_error_vs_x(sigma_list,rel_control_errors,rel_state_errors,rel_adjoint_errors,axis="loglog",x_axis=r'$\sigma$',path=PLOTS+"err_vs_sigma_rel")

    sigma = sigma_og

#============================================================
#%% Do error analysis for different control domains
#============================================================
if DIFFERENT_CONTROL_DOMAINS:
    print("\n" + "="*60)
    print("ERROR ANALYSIS FOR DIFFERENT CONTROL DOMAINS")
    print("="*60)
    subdomains = {
        "full": [((0.0, 1.0), (0.0, 1.0))],
        "central": [((0.25, 0.75), (0.25, 0.75))],
        "disconnected": [((0.0, 0.5), (0.0, 0.5)), ((0.5,1.0),(0.5,1.0))],
        "upwind": [((0.0, 0.5), (0.0, 1.0))],
        "downwind": [((0.5, 1.0), (0.0, 1.0))]
    }
    print(subdomains.keys())
    control_errors = []; rel_control_errors = []
    state_errors = []; rel_state_errors = []
    adjoint_errors = []; rel_adjoint_errors = []

    for omega_str,omega in subdomains.items():
        print("\n" + "-"*60)
        print(omega_str+" control")
        print("-"*60)
        # Initialize FOM
        p.omega = omega
        fom = supplements.parabolic_model(p)
        fom.build_problem()
        opt_FOM = optimization.optimization_class(fom,sigma,tol_abs,tol_rel)
        opt_FOM.Y_d = np.repeat( y_d.reshape(-1,1), m.K, axis=1 )
        opt_FOM.U_d = np.repeat( u_d.reshape(-1,1), m.K, axis=1 )
        
        # Solve FOM
        u_FOM, history_FOM = opt_FOM.solve( U_0, "BB",
                            print_info=True,
                            print_final=True,
                            plot_grad_convergence=True,
                            save_plot_grad_convergence=GENERATE_PLOTS,
                            path=PLOTS+"convergence_FOM_"+omega_str+"_control",
                        )
        U_FOM = fom.vector_to_matrix(u_FOM,option="control")
        Y_FOM = fom.solve_state(U_FOM)
        P_FOM = fom.solve_adjoint(opt_FOM.Y_d-Y_FOM)

        # Get snapshots
        if optimal_snapshots == True: # train with optimal snapshots
            snapshots = [Y_FOM, P_FOM]
            print("\nUsing optimal FOM snapshots for POD basis")
        else: # train with initial snapshots
            snapshots = fom.get_snapshots(U_0,opt_FOM.Y_d)
            print("\nUsing initial snapshots for POD basis")

        # Initialize ROM
        rom = supplements.parabolic_model(p)
        rom.build_problem()
        pod = reduce.pod(rom)
        POD_basis, POD_values = pod.pod_basis(snapshots,l)
        Y_d_ROM, U_d_ROM, U_0_ROM= pod.project(POD_basis,opt_FOM.Y_d,opt_FOM.U_d,U_0)
        opt_ROM = optimization.optimization_class(pod.model,sigma,tol_abs,tol_rel)
        opt_ROM.Y_d = Y_d_ROM
        opt_ROM.U_d = U_d_ROM

        # Solve ROM 
        print("\nSOLVING REDUCED-ORDER MODEL (ROM)")
        u_ROM, history_ROM = opt_ROM.solve( U_0_ROM, "BB",
                                    print_info=True,
                                    print_final=True,
                                    plot_grad_convergence=True,
                                    save_plot_grad_convergence=GENERATE_PLOTS,
                                    path=PLOTS+"convergence_ROM_"+omega_str+"_control",
                            )
        U_ROM = pod.model.vector_to_matrix(u_ROM,option="control")
        Y_ROM = pod.model.solve_state(U_ROM)
        P_ROM = pod.model.solve_adjoint(opt_ROM.Y_d - Y_ROM)

        # Project ROM solutions into full space
        U_ROM_full = POD_basis @ U_ROM  
        Y_ROM_full = POD_basis @ Y_ROM
        P_ROM_full = POD_basis @ P_ROM

        # Compute a-priori (absolute and relative) errors in full space
        control_error = fom.eval_L2H_norm(U_ROM_full - U_FOM, space_norm="control")
        state_error = fom.eval_L2H_norm(Y_ROM_full - Y_FOM, space_norm)
        adjoint_error = fom.eval_L2H_norm(P_ROM_full - P_FOM, space_norm)
        rel_control_error = fom.eval_L2H_norm(U_ROM_full - U_FOM, space_norm="control") / m.eval_L2H_norm(U_FOM, space_norm="control")
        rel_state_error = fom.eval_L2H_norm(Y_ROM_full - Y_FOM, space_norm) / m.eval_L2H_norm(Y_FOM, space_norm)
        rel_adjoint_error = fom.eval_L2H_norm(P_ROM_full - P_FOM, space_norm) / m.eval_L2H_norm(P_FOM, space_norm)

        control_errors.append(control_error); rel_control_errors.append(rel_control_error)
        state_errors.append(state_error); rel_state_errors.append(rel_state_error)
        adjoint_errors.append(adjoint_error); rel_adjoint_errors.append(rel_adjoint_error)
        
        # Compute a-posteriori estimate
        aposteriori, residual = opt_FOM.eval_aposteriori_estimate(U_ROM_full,fom,opt_FOM.Y_d)
        print("\n### ERROR ESTIMATES ###\n")
        print(f"A-posteriori estimate: {aposteriori:.6e}")
        print(f"A-priori estimate: {control_error:.6e}")

        # Print results to console
        print("\n### A-PRIORI ###\n")
        print(f"Control error: {control_error:.6e}")
        print(f"State error: {state_error:.6e}")
        print(f"Adjoint error: {adjoint_error:.6e}")

        print(f"\nRelative control error: {rel_control_error:.6e}")
        print(f"Relative state error: {rel_state_error:.6e}")
        print(f"Relative adjoint error: {rel_adjoint_error:.6e}")

        rom_time = history_ROM['time'] + pod.offline_time_cholesky + pod.offline_time_basisconstruction + pod.offline_time_projection
        print(f"\n###PERFROMANCE###\nFOM optimization time: {history_FOM['time']:.2f} seconds")
        print(f"ROM optimization time: {rom_time:.2f} seconds")
        print(f"Speedup factor: {history_FOM['time']/rom_time:.2f}x")

        if GENERATE_PLOTS:
            print("\nPlotting controls, states, and adjoints...")
            # first inner time step
            fom.plot_3d(U_FOM[:,1], save_png=True, path=PLOTS+f"U_FOM_1_"+omega_str+"_control.png")
            fom.plot_3d(Y_FOM[:,1], save_png=True, path=PLOTS+f"Y_FOM_1_"+omega_str+"_control.png")
            fom.plot_3d(P_FOM[:,1], save_png=True, path=PLOTS+f"P_FOM_1_"+omega_str+"_control.png")
            fom.plot_3d(U_ROM_full[:,1], save_png=True, path=PLOTS+f"U_ROM_1_"+omega_str+"_control.png")
            fom.plot_3d(Y_ROM_full[:,1], save_png=True, path=PLOTS+f"Y_ROM_1_"+omega_str+"_control.png")
            fom.plot_3d(P_ROM_full[:,1], save_png=True, path=PLOTS+f"P_ROM_1_"+omega_str+"_control.png")
            # middle time step
            fom.plot_3d(U_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_FOM_{int((p.K-1)/2)}_"+omega_str+"_control.png")
            fom.plot_3d(Y_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_FOM_{int((p.K-1)/2)}_"+omega_str+"_control.png")
            fom.plot_3d(P_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_FOM_{int((p.K-1)/2)}_"+omega_str+"_control.png")
            fom.plot_3d(U_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_ROM_{int((p.K-1)/2)}_"+omega_str+"_control.png")
            fom.plot_3d(Y_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_ROM_{int((p.K-1)/2)}_"+omega_str+"_control.png")
            fom.plot_3d(P_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_ROM_{int((p.K-1)/2)}_"+omega_str+"_control.png")
            # final inner time step
            fom.plot_3d(U_FOM[:,p.K-2], save_png=True, path=PLOTS+f"U_FOM_{p.K-2}_"+omega_str+"_control.png")
            fom.plot_3d(Y_FOM[:,p.K-2], save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}_"+omega_str+"_control.png")
            fom.plot_3d(P_FOM[:,p.K-2], save_png=True, path=PLOTS+f"P_FOM_{p.K-2}_"+omega_str+"_control.png")
            fom.plot_3d(U_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"U_ROM_{p.K-2}_"+omega_str+"_control.png")
            fom.plot_3d(Y_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"Y_ROM_{p.K-2}_"+omega_str+"_control.png")
            fom.plot_3d(P_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"P_ROM_{p.K-2}_"+omega_str+"_control.png")

    if GENERATE_PLOTS:
        print("\nGenerating error plots...")
        fom.plot_error_vs_x(list(subdomains.keys()),control_errors,state_errors,adjoint_errors,axis="semilogy",path=PLOTS+"err_vs_controldomain_abs")
        fom.plot_error_vs_x(list(subdomains.keys()),rel_control_errors,rel_state_errors,rel_adjoint_errors,axis="semilogy",path=PLOTS+"err_vs_controldomain_rel")

    p.omega = [((0.0, 1.0), (0.0, 1.0))]

#============================================================
#%% Initial vs optimal snapshots
#============================================================
if REDUCED_SPACE_VARIANTS:
    print("\n" + "="*60)
    print("ERROR ANALYSIS FOR INITIAL VS OPTIMAL SNAPSHOTS")
    print("="*60)

    # Initialize FOM
    fom = supplements.parabolic_model(p)
    fom.build_problem()
    opt_FOM = optimization.optimization_class(fom,sigma,tol_abs,tol_rel)
    opt_FOM.Y_d = np.repeat( y_d.reshape(-1,1), m.K, axis=1 )
    opt_FOM.U_d = np.repeat( u_d.reshape(-1,1), m.K, axis=1 )

    # Solve FOM
    print("\nSOLVING FULL-ORDER MODEL (FOM)")
    u_FOM, history_FOM = opt_FOM.solve( U_0, "BB",
                            print_info=True,
                            print_final=True,
                            plot_grad_convergence=True,
                            save_plot_grad_convergence=GENERATE_PLOTS,
                            path=PLOTS+"convergence_FOM",
                        )
    U_FOM = fom.vector_to_matrix(u_FOM,option="control")
    Y_FOM = fom.solve_state(U_FOM)
    P_FOM = fom.solve_adjoint(opt_FOM.Y_d-Y_FOM)

    # Get snapshots
    snapshot_set = {
        "initial": fom.get_snapshots(U_0,opt_FOM.Y_d), 
        "optimal":[Y_FOM, P_FOM]
    }

    # Lists to track errors
    control_errors_initial = []
    state_errors_initial = []
    adjoint_errors_initial = []
    control_errors_optimal = []
    state_errors_optimal = []
    adjoint_errors_optimal = []
    max_l_initial = None
    max_l_optimal = None

    for snaps_str,snaps in snapshot_set.items():
        print("\n" + "-"*60)
        print(snaps_str+" snapshots")
        print("-"*60)
        # Initialize ROM
        rom = supplements.parabolic_model(p)
        rom.build_problem()
        pod = reduce.pod(rom)
        POD_basis, POD_values = pod.pod_basis(snaps,l)
        Y_d_ROM, U_d_ROM, U_0_ROM= pod.project(POD_basis,opt_FOM.Y_d,opt_FOM.U_d,U_0)
        opt_ROM = optimization.optimization_class(pod.model,sigma,tol_abs,tol_rel)
        opt_ROM.Y_d = Y_d_ROM
        opt_ROM.U_d = U_d_ROM

        if snaps_str == "initial":
            max_l_initial = pod.basissize
            POD_values_initial = POD_values
            POD_values_initial_normalized = pod.POD_values_normalized
        elif snaps_str == "optimal":
            max_l_optimal = pod.basissize

        # Solve ROM 
        print("\nSOLVING REDUCED-ORDER MODEL (ROM)")
        u_ROM, _ = opt_ROM.solve( U_0_ROM, "BB",
                                    print_info=True,
                                    print_final=True,
                                    plot_grad_convergence=True,
                                    save_plot_grad_convergence=GENERATE_PLOTS,
                                    path=PLOTS+"convergence_ROM_"+snaps_str+"_snapshots",
                            )
        U_ROM = pod.model.vector_to_matrix(u_ROM,option="control")
        Y_ROM = pod.model.solve_state(U_ROM)
        P_ROM = pod.model.solve_adjoint(opt_ROM.Y_d - Y_ROM)

        # Vary POD basis rank
        control_errors_l = []
        state_errors_l = []
        adjoint_errors_l = []

        for j in range(1,pod.basissize+1):
            print("\n" + "-"*60)
            print("Number of snapshots l="+str(j))
            print("-"*60)
            rom_l = supplements.parabolic_model(p)
            rom_l.build_problem()
            pod_l = reduce.pod(rom_l)

            POD_basis_l, POD_values_l = pod_l.pod_basis(snaps,j)
            Y_d_l, U_d_l, U_0_ROM_l = pod_l.project(POD_basis_l,opt_FOM.Y_d,opt_FOM.U_d,U_0)
            opt_ROM_l = optimization.optimization_class(pod_l.model,sigma,tol_abs,tol_rel)
            opt_ROM_l.Y_d = Y_d_l
            opt_ROM_l.U_d = U_d_l

            # Solve ROM 
            print("\nSOLVING REDUCED-ORDER MODEL (ROM)")

            u_ROM_l, history_ROM_l = opt_ROM_l.solve( U_0_ROM_l, "BB",
                                        print_info=True,
                                        print_final=True,
                                        plot_grad_convergence=True,
                                        save_plot_grad_convergence=GENERATE_PLOTS,
                                        path=PLOTS+"convergence_ROM_"+str(j)+"_snapshots_"+snaps_str,
                                )
            U_ROM_l = pod_l.model.vector_to_matrix(u_ROM_l,option="control")
            Y_ROM_l = pod_l.model.solve_state(U_ROM_l)
            P_ROM_l = pod_l.model.solve_adjoint(opt_ROM_l.Y_d - Y_ROM_l)
            rom_time = history_ROM_l['time'] + pod_l.offline_time_cholesky + pod_l.offline_time_basisconstruction + pod_l.offline_time_projection
            print(f"ROM optimization time (offline+solving): {rom_time:.3f} seconds")

            # Recover FOM solution from ROM
            U_ROM_full_l = POD_basis_l @ U_ROM_l  # Project control back to full space
            Y_ROM_full_l = POD_basis_l @ Y_ROM_l
            P_ROM_full_l = POD_basis_l @ P_ROM_l

            # Compute error
            control_error = fom.eval_L2H_norm(U_ROM_full_l - U_FOM, space_norm="control")
            state_error = fom.eval_L2H_norm(Y_ROM_full_l - Y_FOM, space_norm)
            adjoint_error = fom.eval_L2H_norm(P_ROM_full_l - P_FOM, space_norm)

            control_errors_l.append(control_error)
            state_errors_l.append(state_error)
            adjoint_errors_l.append(adjoint_error)

            print(f"Control error: {control_error:.6e}")
            print(f"State error: {state_error:.6e}")
            print(f"Adjoint error: {adjoint_error:.6e}")
        
        if snaps_str == "initial":
            control_errors_initial = control_errors_l
            state_errors_initial = state_errors_l
            adjoint_errors_initial = adjoint_errors_l
        elif snaps_str == "optimal":
            control_errors_optimal = control_errors_l
            state_errors_optimal = state_errors_l
            adjoint_errors_optimal = adjoint_errors_l

        # Print errors vs POD rank
        l_list = range(1,pod.basissize+1)
        print("\nControl error || u_POD - u_FE || depending on l:")
        for i in l_list:
            print("l="+str(i)+", err="+str(control_errors_l[i-1]))
        print("\nState error || y_POD - y_FE || depending on l:")
        for i in l_list:
            print("l="+str(i)+", err="+str(state_errors_l[i-1]))
        print("\nAdjoint error || p_POD - p_FE || depending on l:")
        for i in l_list:
            print("l="+str(i)+", err="+str(adjoint_errors_l[i-1]))

        if GENERATE_PLOTS:
            pod.plot_pod_values(path=PLOTS+snaps_str+"_snapshots")
            fom.plot_error_vs_x(list(map(str,l_list)),control_errors_l,state_errors_l,adjoint_errors_l,axis="semilogy",x_axis="POD rank",path=PLOTS+"err_vs_PODrank_abs_"+snaps_str+"_snapshots")
    
    if GENERATE_PLOTS:
        ranks = range(1,max(max_l_initial,max_l_optimal)+1)
        fom.plot_error_vs_x(ranks,control_errors_initial,state_errors_initial,adjoint_errors_initial,control_errors_optimal,state_errors_optimal,adjoint_errors_optimal,firstgroup="init ",secondgroup="opt ",axis="semilogy",x_axis="POD rank",path=PLOTS+"err_vs_snaps")
        pod.plot_pod_values(path=PLOTS,otherpodvalues=POD_values_initial,otherpodvalues_normalized=POD_values_initial_normalized)

#============================================================
#%% Compare different gammas and vary peclet number
#============================================================
if DIFFERENT_PDE_PARAMETERS:
    print("\n" + "="*60)
    print("ERROR ANALYSIS FOR DIFFERENT PDE PARAMETERS")
    print("="*60)

    for compare in ["peclet", "reaction"]:
        print("\n" + "-"*60)
        print("comparing for different "+compare)
        print("-"*60)
        
        if compare == "peclet":
            pde_parameters = {
                "small_peclet":{
                    "kappa": 1.0,
                    "beta": [1.0,0.0],
                    "gamma": 1.0
                },
                "balanced_peclet":{
                    "kappa": 0.1,
                    "beta": [2.0,0.0],
                    "gamma": 1.0
                },
                "large_peclet":{
                    "kappa": 0.01,
                    "beta": [4.0,0.0],
                    "gamma": 1.0
                },
            }
            labels = ["0.05","1","20"]
            xlabel = "Péclet number"
        elif compare == "reaction":
            pde_parameters = {
                "big_negative_reaction":{
                    "kappa": 0.1,
                    "beta": [2.0,0.0],
                    "gamma": -5.0
                },
                "negative_reaction":{
                    "kappa": 0.1,
                    "beta": [2.0,0.0],
                    "gamma": -1.0
                },
                "positive_reaction":{
                    "kappa": 0.1,
                    "beta": [2.0,0.0],
                    "gamma": 1.0
                },
                "big_positive_reaction":{
                    "kappa": 0.1,
                    "beta": [2.0,0.0],
                    "gamma": 5.0
                }
            }
            labels = ["-5","-1","1","5"]
            xlabel = "$\gamma$"
        
        control_errors = []; rel_control_errors = []
        state_errors = []; rel_state_errors = []
        adjoint_errors = []; rel_adjoint_errors = []

        for param_str,param in pde_parameters.items():
            print("\n" + "-"*60)
            print("pde setting: "+param_str)
            print("-"*60)
            # Initialize FOM
            p.kappa = fenics.Constant(pde_parameters[param_str]["kappa"])
            p.beta_x1 = fenics.Constant(pde_parameters[param_str]["beta"][0])
            p.beta_x2 = fenics.Constant(pde_parameters[param_str]["beta"][1])
            p.beta = fenics.as_vector((p.beta_x1, p.beta_x2))
            p.gamma = fenics.Constant(pde_parameters[param_str]["gamma"])
            fom = supplements.parabolic_model(p)
            fom.build_problem()
            opt_FOM = optimization.optimization_class(fom,sigma,tol_abs,tol_rel)
            opt_FOM.Y_d = np.repeat( y_d.reshape(-1,1), m.K, axis=1 )
            opt_FOM.U_d = np.repeat( u_d.reshape(-1,1), m.K, axis=1 )
            
            # Solve FOM
            u_FOM, history_FOM = opt_FOM.solve( U_0, "BB",
                                print_info=True,
                                print_final=True,
                                plot_grad_convergence=True,
                                save_plot_grad_convergence=GENERATE_PLOTS,
                                path=PLOTS+"convergence_FOM_"+param_str,
                            )
            U_FOM = fom.vector_to_matrix(u_FOM,option="control")
            Y_FOM = fom.solve_state(U_FOM)
            P_FOM = fom.solve_adjoint(opt_FOM.Y_d-Y_FOM)

            # Get snapshots
            if optimal_snapshots == True: # train with optimal snapshots
                snapshots = [Y_FOM, P_FOM]
                print("\nUsing optimal FOM snapshots for POD basis")
            else: # train with initial snapshots
                snapshots = fom.get_snapshots(U_0,opt_FOM.Y_d)
                print("\nUsing initial snapshots for POD basis")

            # Initialize ROM
            rom = supplements.parabolic_model(p)
            rom.build_problem()
            pod = reduce.pod(rom)
            POD_basis, POD_values = pod.pod_basis(snapshots,l)
            Y_d_ROM, U_d_ROM, U_0_ROM= pod.project(POD_basis,opt_FOM.Y_d,opt_FOM.U_d,U_0)
            opt_ROM = optimization.optimization_class(pod.model,sigma,tol_abs,tol_rel)
            opt_ROM.Y_d = Y_d_ROM
            opt_ROM.U_d = U_d_ROM

            # Solve ROM 
            print("\nSOLVING REDUCED-ORDER MODEL (ROM)")
            u_ROM, history_ROM = opt_ROM.solve( U_0_ROM, "BB",
                                        print_info=True,
                                        print_final=True,
                                        plot_grad_convergence=True,
                                        save_plot_grad_convergence=GENERATE_PLOTS,
                                        path=PLOTS+"convergence_ROM_"+param_str,
                                )
            U_ROM = pod.model.vector_to_matrix(u_ROM,option="control")
            Y_ROM = pod.model.solve_state(U_ROM)
            P_ROM = pod.model.solve_adjoint(opt_ROM.Y_d - Y_ROM)

            # Project ROM solutions into full space
            U_ROM_full = POD_basis @ U_ROM  
            Y_ROM_full = POD_basis @ Y_ROM
            P_ROM_full = POD_basis @ P_ROM

            # Compute (absolute and relative) errors in full space
            control_error = fom.eval_L2H_norm(U_ROM_full - U_FOM, space_norm="control")
            state_error = fom.eval_L2H_norm(Y_ROM_full - Y_FOM, space_norm)
            adjoint_error = fom.eval_L2H_norm(P_ROM_full - P_FOM, space_norm)
            rel_control_error = fom.eval_L2H_norm(U_ROM_full - U_FOM, space_norm="control") / m.eval_L2H_norm(U_FOM, space_norm="control")
            rel_state_error = fom.eval_L2H_norm(Y_ROM_full - Y_FOM, space_norm) / m.eval_L2H_norm(Y_FOM, space_norm)
            rel_adjoint_error = fom.eval_L2H_norm(P_ROM_full - P_FOM, space_norm) / m.eval_L2H_norm(P_FOM, space_norm)

            control_errors.append(control_error); rel_control_errors.append(rel_control_error)
            state_errors.append(state_error); rel_state_errors.append(rel_state_error)
            adjoint_errors.append(adjoint_error); rel_adjoint_errors.append(rel_adjoint_error)
            
            # Compute a-posteriori estimate
            aposteriori, residual = opt_FOM.eval_aposteriori_estimate(U_ROM_full,fom,opt_FOM.Y_d)
            print("\n### ERROR ESTIMATES ###\n")
            print(f"A-posteriori estimate: {aposteriori:.6e}")
            print(f"A-priori estimate: {control_error:.6e}")

            # Print results to console
            print("\n### A-PRIORI ###\n")
            print(f"Control error: {control_error:.6e}")
            print(f"State error: {state_error:.6e}")
            print(f"Adjoint error: {adjoint_error:.6e}")

            print(f"\nRelative control error: {rel_control_error:.6e}")
            print(f"Relative state error: {rel_state_error:.6e}")
            print(f"Relative adjoint error: {rel_adjoint_error:.6e}")

            rom_time = history_ROM['time'] + pod.offline_time_cholesky + pod.offline_time_basisconstruction + pod.offline_time_projection
            print(f"\nFOM optimization time: {history_FOM['time']:.2f} seconds")
            print(f"ROM optimization time: {rom_time:.2f} seconds")
            print(f"Speedup factor: {history_FOM['time']/rom_time:.2f}x")

            if GENERATE_PLOTS:
                print("\nPlotting controls, states, and adjoints...")
                # first inner time step
                fom.plot_3d(U_FOM[:,1], save_png=True, path=PLOTS+f"U_FOM_1_"+param_str+".png")
                fom.plot_3d(Y_FOM[:,1], save_png=True, path=PLOTS+f"Y_FOM_1_"+param_str+".png")
                fom.plot_3d(P_FOM[:,1], save_png=True, path=PLOTS+f"P_FOM_1_"+param_str+"png")
                fom.plot_3d(U_ROM_full[:,1], save_png=True, path=PLOTS+f"U_ROM_1_"+param_str+".png")
                fom.plot_3d(Y_ROM_full[:,1], save_png=True, path=PLOTS+f"Y_ROM_1_"+param_str+".png")
                fom.plot_3d(P_ROM_full[:,1], save_png=True, path=PLOTS+f"P_ROM_1_"+param_str+".png")
                # middle time step
                fom.plot_3d(U_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_FOM_{int((p.K-1)/2)}_"+param_str+".png")
                fom.plot_3d(Y_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_FOM_{int((p.K-1)/2)}_"+param_str+".png")
                fom.plot_3d(P_FOM[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_FOM_{int((p.K-1)/2)}_"+param_str+".png")
                fom.plot_3d(U_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_ROM_{int((p.K-1)/2)}_"+param_str+".png")
                fom.plot_3d(Y_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_ROM_{int((p.K-1)/2)}_"+param_str+".png")
                fom.plot_3d(P_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_ROM_{int((p.K-1)/2)}_"+param_str+".png")
                # final inner time step
                fom.plot_3d(U_FOM[:,p.K-2], save_png=True, path=PLOTS+f"U_FOM_{p.K-2}_"+param_str+".png")
                fom.plot_3d(Y_FOM[:,p.K-2], save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}_"+param_str+".png")
                fom.plot_3d(P_FOM[:,p.K-2], save_png=True, path=PLOTS+f"P_FOM_{p.K-2}_"+param_str+".png")
                fom.plot_3d(U_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"U_ROM_{p.K-2}_"+param_str+".png")
                fom.plot_3d(Y_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"Y_ROM_{p.K-2}_"+param_str+".png")
                fom.plot_3d(P_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"P_ROM_{p.K-2}_"+param_str+".png")

        if GENERATE_PLOTS:
            print("\nGenerating error plots...")
            fom.plot_error_vs_x(labels,control_errors,state_errors,adjoint_errors,axis="semilogy",x_axis=xlabel,path=PLOTS+"err_vs_pdeparams_"+compare+"_abs")
            fom.plot_error_vs_x(labels,rel_control_errors,rel_state_errors,rel_adjoint_errors,axis="semilogy",x_axis=xlabel,path=PLOTS+"err_vs_pdeparams_"+compare+"_rel")
            fom.plot_error_vs_x(labels,control_errors,state_errors,adjoint_errors,rel_control_errors,rel_state_errors,rel_adjoint_errors,axis="semilogy",x_axis=xlabel,path=PLOTS+"err_vs_pdeparams_"+compare)

    
    p.kappa = fenics.Constant(1.0)
    p.beta_x1 = fenics.Constant(1.0)
    p.beta_x2 = fenics.Constant(0.0)
    p.beta = fenics.as_vector((p.beta_x1, p.beta_x2))
    p.gamma = fenics.Constant(1.0)

#============================================================
#%% Plots
#============================================================
if GENERATE_PLOTS and SOLVE_FOM and SOLVE_ROM:
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    '''
    for k in range(0, p.K, 10):
        print(f"Plotting time step {k}...")
        m.plot_3d(Y_opt[:,k], save_png=True, path=PLOTS+f"Y_FOM_{k}.png") # title=f"FOM State t={k}"
        m.plot_3d(U_opt[:,k], save_png=True, path=PLOTS+f"U_FOM_{k}.png") # title=f"FOM Control t={k}"
        m.plot_3d(P_opt[:,k], save_png=True, path=PLOTS+f"P_FOM_{k}.png") # title=f"FOM Adjoint t={k}"
        m.plot_3d(Y_ROM_full[:,k], save_png=True, path=PLOTS+f"Y_ROM_{k}.png") # title=f"ROM State t={k}"
        m.plot_3d(U_ROM_full[:,k], save_png=True, path=PLOTS+f"U_ROM_{k}.png") # title=f"ROM Control t={k}"
        m.plot_3d(P_ROM_full[:,k], save_png=True, path=PLOTS+f"P_ROM_{k}.png") # title=f"ROM Adjoint t={k}"
    '''
    # First inner time step
    m.plot_3d(Y_opt[:,1], save_png=True, path=PLOTS+"Y_FOM_1.png") 
    m.plot_3d(U_opt[:,1], save_png=True, path=PLOTS+"U_FOM_1.png") 
    m.plot_3d(P_opt[:,1], save_png=True, path=PLOTS+"P_FOM_1.png") 
    m.plot_3d(Y_ROM_full[:,1], save_png=True, path=PLOTS+"Y_ROM_1.png") 
    m.plot_3d(U_ROM_full[:,1], save_png=True, path=PLOTS+"U_ROM_1.png") 
    m.plot_3d(P_ROM_full[:,1], save_png=True, path=PLOTS+"P_ROM_1.png") 

    # Middle time step
    m.plot_3d(Y_opt[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_FOM_{int((p.K-1)/2)}.png") 
    m.plot_3d(U_opt[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_FOM_{int((p.K-1)/2)}.png") 
    m.plot_3d(P_opt[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_FOM_{int((p.K-1)/2)}.png") 
    m.plot_3d(Y_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"Y_ROM_{int((p.K-1)/2)}.png") 
    m.plot_3d(U_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"U_ROM_{int((p.K-1)/2)}.png") 
    m.plot_3d(P_ROM_full[:,int((p.K-1)/2)], save_png=True, path=PLOTS+f"P_ROM_{int((p.K-1)/2)}.png") 

    # Final inner time step
    m.plot_3d(Y_opt[:,p.K-2], save_png=True, path=PLOTS+f"Y_FOM_{p.K-2}.png") 
    m.plot_3d(U_opt[:,p.K-2], save_png=True, path=PLOTS+f"U_FOM_{p.K-2}.png") 
    m.plot_3d(P_opt[:,p.K-2], save_png=True, path=PLOTS+f"P_FOM_{p.K-2}.png") 
    m.plot_3d(Y_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"Y_ROM_{p.K-2}.png") 
    m.plot_3d(U_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"U_ROM_{p.K-2}.png") 
    m.plot_3d(P_ROM_full[:,p.K-2], save_png=True, path=PLOTS+f"P_ROM_{p.K-2}.png") 
    
    print("All plots saved to " + PLOTS)

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)