"""
optimization.py
@author: Thanh-Van Huynh
solve_BB() from Andrea Petrocchi (July 2023)
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

class optimization_class():
    def __init__( self,m,sigma,tol_abs,tol_rel):
        self.m = m
        self.Y_d = 0. # desired observable
        self.U_d = 0. # control energy
        self.sigma = sigma # regularization factor in the cost functional
        self.tol_abs = tol_abs; self.tol_rel = tol_rel # tolerance for the optimization algorithm
        self.prec = "id" # standard preconditioner is the identity

    def eval_cost( self, u ):
        U = self.m.vector_to_matrix(u, option="control")
        Z = self.m.solve_state(U) - self.Y_d
        W = U - self.U_d
        F = 0.5 * self.m.eval_L2H_prod(Z,Z) + 0.5 * self.sigma * self.m.eval_L2H_prod(W,W)
        return F

    def eval_grad( self, u ):
        U = self.m.vector_to_matrix(u, option="control")
        Y = self.m.solve_state(U)
        Z = self.Y_d - Y
        P = self.m.solve_adjoint(Z)
        BP = self.eval_Bp(P)
        J = self.sigma * ( U - self.U_d ) - BP
        return J.flatten(), Y, P
    
    def eval_Bp(self, P):
        grad_from_adjoint = np.zeros((self.m.control_dof, self.m.K))
        for k in range(self.m.K):
            if self.m.is_reduced:
                grad_from_adjoint[:, k] = self.m.chi @ P[:, k]
            else:
                grad_from_adjoint[:, k] = self.m.chi * P[:, k]
        return grad_from_adjoint
        
    def solve( self, U_0, method, prec="id", print_info=False, print_final=False, plot_grad_convergence=False, save_plot_grad_convergence=False, path=None ):
        options = {"print_info": print_info,
                   "print_final": print_final,
                   "plot_grad_convergence":plot_grad_convergence,
                   "save_plot_grad_convergence":save_plot_grad_convergence,
                   "path":path}
        if method == "BB":
            u_opt, history = self.solve_BB( U_0.flatten(), options )
        return u_opt, history
    
    #%% Optimization algorithms
    
    def solve_BB( self, u_0, options ):
        # Initialize
        t = time()
        u_km1 = np.zeros_like(u_0)
        grad_km1, Y, P = self.eval_grad(u_km1)
        u_k = grad_km1
        grad_k, Y, P = self.eval_grad(u_k)
        k = 0
        err = self.m.eval_L2H_norm(grad_k)
        tol = self.tol_rel * err + self.tol_abs
        history = {"gradient": [grad_k],
                   "error": [err],
                   "cost": []
                   }
        
        if options["print_info"]:
            cost = self.eval_cost(u_k); 
            history["cost"].append(cost)
            print("k:{}, cost={}, err={}".format(k,cost,err))
            
        while err > tol and k<100 :
            # Compute BB steplength
            sk = u_k - u_km1
            dk = grad_k - grad_km1
            skdk = self.m.eval_L2H_prod(sk,dk)
            if k%2==0: # k is even
                alpha_k = self.m.eval_L2H_prod(dk,dk)/skdk
            else: # k is odd
                alpha_k = skdk / self.m.eval_L2H_prod(sk,sk)

            # Update
            u_km1 = u_k.copy()
            u_k = u_k - grad_k/alpha_k
            grad_km1 = grad_k.copy()

            # Compute new gradient and its norm
            grad_k, Y, P = self.eval_grad(u_k)
            err = self.m.eval_L2H_norm(grad_k)
            
            # Update history
            history["gradient"].append(grad_k)
            history["error"].append(err)
            k += 1

            if options["print_info"]:
                cost = self.eval_cost(u_k); 
                history["cost"].append(cost)
                print("k:{}, cost={}, err={}".format(k,cost,err))
        
        # Finalize
        t = time() - t
        history["Y_opt"] = Y
        history["P_opt"] = P
        history["time"] = t
        history["k"] = k
                
        if options["print_final"]:
            print("Converged in k={} iterations. Time: {}".format(k,t))
        if options["plot_grad_convergence"]:
            plt.figure(figsize=(8, 6)) 
            plt.semilogy(history["error"])
            plt.xlabel("Iteration")
            plt.ylabel(r"$||\nabla\hat{J}(u_k)||_U$")
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.grid(True, alpha=0.3, ls="--")
            plt.tight_layout()
            # plt.title(r"BB. Convergence of $\|\nabla F(u_k)\|_U$")
            if options["save_plot_grad_convergence"]:
                plt.savefig(options["path"]+"_gradient",dpi=600)
            plt.close()
        if len(history["cost"])>1:
            plt.figure(figsize=(8, 6))
            plt.semilogy(history["cost"])
            plt.xlabel("Iteration")
            plt.ylabel(r"$||\hat{J}(u_k)||_U$")
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.grid(True, alpha=0.3, ls="--")
            plt.tight_layout()
            # plt.title(r"BB. Convergence of $\|F(u_k)\|_U$")
            if options["save_plot_grad_convergence"]:
                plt.savefig(options["path"]+"_cost",dpi=600)
            plt.close()
        return u_k, history
    
    def eval_aposteriori_estimate(self, U_ROM_full, fom_model, Y_d_FOM):
        # Consider control domain
        chi = fom_model.chi
        U_ROM_full = chi[:, None] * U_ROM_full

        # Get residual/gradient given by sigma*U-BP
        Y_ROM_full = fom_model.solve_state(U_ROM_full)
        P_ROM_full = fom_model.solve_adjoint(Y_d_FOM - Y_ROM_full)
        BP_full = chi[:, None] * P_ROM_full
        residual = self.sigma * U_ROM_full - BP_full

        # Get norm
        estimate = fom_model.eval_L2H_norm(residual, space_norm="L2")/self.sigma

        return estimate, residual