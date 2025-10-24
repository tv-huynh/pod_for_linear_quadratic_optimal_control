"""
ES6: main
"""

import fenics
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized
import supplements
import optimization
import reduce
import os

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

# TEMP = "temp/"; m.format_folder(TEMP)
if GENERATE_PLOTS:
    PLOTS = "plots/"; m.format_folder(PLOTS)


#%% Solve FOM (Barzilai-Borwein)
u_opt_BB, history = opt.solve( U_0, "BB",
                        print_info=True,
                        print_final=True,
                        plot_grad_convergence=True,
                        save_plot_grad_convergence=GENERATE_PLOTS,
                        path=PLOTS+"convergence_BB",
                    )
U_opt_BB = m.vector_to_matrix(u_opt_BB)
Y_opt_BB = m.solve_state(U_opt_BB)
P_opt_BB = m.solve_adjoint(opt.Y_d-Y_opt_BB)


#%% Construct ROM out of optimal FOM snapshots
l = 20
optimal_snapshots = True

if optimal_snapshots == True: # train with optimal snapshots
    snapshots = [Y_opt_BB, P_opt_BB]
else: # train with initial snapshots
    snapshots = [m.get_snapshots(U_0,opt.Y_d)]

pod = reduce.pod(m)
POD_Basis, POD_values = pod.pod_basis(snapshots,l)
pod.project(POD_Basis)

opt_ROM = optimization.optimization_class(pod.model)

# project the desired states/control onto the reduced basis
opt_ROM.Y_d = pod.POD_Basis.T @ opt.Y_d       # shape (l, K)
opt_ROM.U_d = pod.POD_Basis.T @ opt.U_d       # shape (l, K)

# Project initial guess
U_0_ROM = POD_Basis.T @ U_0          # shape (l, K)

u_BB_ROM, history_BB_ROM = opt_ROM.solve( U_0_ROM, "BB",
                            print_info=True,
                            print_final=True,
                            plot_grad_convergence=True,
                            save_plot_grad_convergence=GENERATE_PLOTS,
                            path=PLOTS+"convergence_BB",
                    )
U_BB_ROM = pod.model.vector_to_matrix(u_BB_ROM)
Y_BB_ROM = pod.model.solve_state(U_BB_ROM)
P_BB_ROM = pod.model.solve_adjoint(opt_ROM.Y_d-Y_BB_ROM)

# Recover FOM solution from ROM
U_BB_ROM = POD_Basis @ U_BB_ROM   # shape (900, K)
Y_BB_ROM = POD_Basis @ Y_BB_ROM   # shape (900, K)
P_BB_ROM = POD_Basis @ P_BB_ROM   # shape (900, K)

if GENERATE_PLOTS:
    for k in range( 0, p.K, 10 ):
        m.plot_3d( Y_opt_BB[:,k], save_png=True, path=PLOTS+"Y_FOM_"+str(k)+".png")
        m.plot_3d( U_opt_BB[:,k], save_png=True, path=PLOTS+"U_FOM_"+str(k)+".png")
        m.plot_3d( P_opt_BB[:,k], save_png=True, path=PLOTS+"P_FOM_"+str(k)+".png")
        m.plot_3d( Y_BB_ROM[:,k], save_png=True, path=PLOTS+"Y_ROM_"+str(k)+".png")
        m.plot_3d( U_BB_ROM[:,k], save_png=True, path=PLOTS+"U_ROM_"+str(k)+".png")
        m.plot_3d( P_BB_ROM[:,k], save_png=True, path=PLOTS+"P_ROM_"+str(k)+".png")
    m.plot_3d( Y_opt_BB[:,p.K-2], save_png=True, path=PLOTS+"Y_FOM_"+str(p.K-2)+".png")
    m.plot_3d( U_opt_BB[:,p.K-2], save_png=True, path=PLOTS+"U_FOM_"+str(p.K-2)+".png")
    m.plot_3d( P_opt_BB[:,p.K-2], save_png=True, path=PLOTS+"p_FOM_"+str(p.K-2)+".png")
    m.plot_3d( Y_BB_ROM[:,p.K-2], save_png=True, path=PLOTS+"Y_ROM_"+str(p.K-2)+".png")
    m.plot_3d( U_BB_ROM[:,p.K-2], save_png=True, path=PLOTS+"U_ROM_"+str(p.K-2)+".png")
    m.plot_3d( P_BB_ROM[:,p.K-2], save_png=True, path=PLOTS+"P_ROM_"+str(p.K-2)+".png")