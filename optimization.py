"""
ES6: optimization
Original work by: Andrea Petrocchi (July 2023)
Modifications and additions by: Thanh-Van Huynh
"""

import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity, csc_matrix, block_diag
from scipy.sparse.linalg import spsolve
from time import time

class optimization_class():
    def __init__( self,m,beta,tol):
        self.m = m
        self.Y_d = 0. # desired observable
        self.U_d = 0. # control energy
        self.beta = beta # regularization factor in the cost functional
        self.tol = tol # tolerance for the optimization algorithm
        self.prec = "id" # standard preconditioner is the identity
    
    def eval_cost( self, u ):
        U = self.m.vector_to_matrix(u, option="control")
        Z = self.m.solve_state(U) - self.Y_d
        W = U - self.U_d
        F = 0.5 * self.m.eval_L2H_prod(Z,Z) + 0.5 * self.beta * self.m.eval_L2H_prod(W,W)
        return F


    def eval_grad( self, u ):
        U = self.m.vector_to_matrix(u, option="control")
        Y = self.m.solve_state(U)
        Z = Y - self.Y_d
        P = self.m.solve_adjoint(Z)
        J = self.beta * ( U - self.U_d ) + P
        return J.flatten(), Y, P

    
    def eval_Qu( self, u ):
        U = self.m.vector_to_matrix(u,option="control")
        Y = self.m.solve_state( U )
        P = self.m.solve_adjoint( Y )
        
        grad_from_adjoint = np.zeros_like(U)
        for k in range(self.m.K):
            grad_from_adjoint[:, k] = self.m.B.T @ P[:, k]
        
        out = self.beta * U + grad_from_adjoint
        return out.flatten()
    
    def eval_b( self ):
        P = self.m.solve_adjoint( self.Y_d )
        
        grad_from_adjoint = np.zeros((self.m.control_dof, self.m.K))
        for k in range(self.m.K):
            grad_from_adjoint[:, k] = self.m.B.T @ P[:, k]
        
        out = self.beta * self.U_d + grad_from_adjoint
        return out.flatten()
        
    def solve( self, U_0, method, prec='id', print_info=False, print_final=False, plot_grad_convergence=False, save_plot_grad_convergence=False, path=None ):
        options = {'print_info': print_info,
                   'print_final': print_final,
                   'plot_grad_convergence':plot_grad_convergence,
                   'save_plot_grad_convergence':save_plot_grad_convergence,
                   'path':path}
        if method == "precCG":
            u_opt = self.solve_precCG( U_0.flatten(), prec, options)
            history = None
        elif method == "BB":
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
        history = {"gradient": [grad_k],
                   "error": [err],
                   "cost": []
                   }
        
        if options['print_info']:
            cost = self.eval_cost(u_k); list_cost = [ cost ]
            print("k:{}, cost={}, err={}".format(k,cost,err))
            
        while err > self.tol and k<500 :
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

            if options['print_info']:
                cost = self.eval_cost(u_k); 
                history["cost"].append(cost)
                print("k:{}, cost={}, err={}".format(k,cost,err))
        
        # Finalize
        t = time() - t
        history["Y_opt"] = Y
        history["P_opt"] = P
        history["time"] = t
        history["k"] = k
                
        if options['print_final']:
            print("Converged in k={} iterations. Time: {}".format(k,t))
        if options['plot_grad_convergence']:
            plt.figure(figsize=(8, 6)) 
            plt.semilogy(history["error"])
            plt.title(r'BB. Convergence of $\|\nabla F(u_k)\|_U$')
            if options['save_plot_grad_convergence']:
                plt.savefig( options['path'] )
            plt.close()
        if len(list_cost)>1:
            plt.semilogy(list_cost)
            plt.title(r'BB. Convergence of $\|F(u_k)\|_U$')
            if options['save_plot_grad_convergence']:
                plt.savefig( options['path']+"_cost" )
            plt.close()
        return u_k, history
    
    
    def solve_precCG( self, u_0, prec, options ):
        t = time()
        # P_inv = self.select_preconditioner(prec)
        u = u_0.copy()
        A = lambda x: self.eval_Qu(x)
        b = self.eval_b()
        k = 0
        
        r = b-A(u)
        z = r.copy() # z = P_inv(r)
        p = z
        err = np.sqrt(np.vdot(r,z)) # r.T.dot( self.m.M.dot(z) )
        list_err = [err]
        if options['print_info']:
            print( "k:{}. Residual norm={}. Error={}".format(
                k, np.sqrt(np.vdot(r,r)), err) )
        
        while True:
            r_old = r.copy()
            z_old = z.copy()
            Ap = A(p)
            alpha = np.vdot(r,z) / ( p.T.dot(Ap) )
            u += alpha*p
            r -= alpha*Ap
            z = r.copy() # z = P_inv(r)
            err = np.sqrt(np.vdot(r,z)) # r.T.dot(self.m.M.dot(z))
            list_err.append(err)
            if err < self.tol or k>200:
                break
            beta = np.vdot(r,z) / np.vdot(r_old,z_old)
            p = z + beta * p
            k += 1
            if options['print_info']:
                print( "k:{}. Residual norm={}. Error={}".format(
                    k, np.sqrt(np.vdot(r,r)), err) )
        t = time() - t
        
        if options['print_final']:
            print("Converged in k={} iterations. Time: {}".format(k,t))
        if options['plot_grad_convergence']:
            plt.semilogy(list_err)
            plt.title(r'precCG. Convergence of $\|\nabla F(u_k)\|_U$')
            if options['save_plot_grad_convergence']:
                plt.savefig( options['path'] )
            plt.figure()
        return u