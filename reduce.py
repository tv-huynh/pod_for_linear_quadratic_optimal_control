"""
reduce.py
@author: Thanh-Van Huynh
pod_basis() from Michael Kartmann (https://github.com/michikartmann/pod_for_linear_quadratic_optimal_control)
"""
import scipy.sparse as sps
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized
from time import time

class pod():
    def __init__(self, model, space_product="L2"):
        self.model = model
        self.space_norm = space_product
        if space_product == "L2":
            S = model.M
        elif space_product == "H10":
            S = model.A
        elif space_product == "H1":
            S = model.M + model.A  
        start_time = time()
        if hasattr(S, "todense"):
            self.W_chol = linalg.cholesky(S.todense())
        else:
            self.W_chol = linalg.cholesky(S)
        end_time = time()
        self.offline_time_cholesky = end_time-start_time

    def pod_basis(self, Y, l, W = None, D = None, flag = 0):
        """
        #     Compute POD basis

        #     Parameters
        #     ----------
        #     Y: list of/or ndarray shape (n_x,n_t),
        #         Matrix containing the vectors {y^k} (Y = [y^1,...,y^nt]),
        #         or a list containing different snapshot matrices
        #     l: int,
        #         Length of the POD-basis.
        #     W: ndarray, shape (n_x,n_x)
        #         Gramian of the Hilbert space X, that containts the snapshots.
        #     D: list of/or ndarray of shape (n_t,n_t)
        #         Matrix containing the weights of the time discretization.
        #     flag: int
        #         parameter deciding which method to use for computing the POD-basis
        #         (if flag==0 svd, flag == 1 eig of YY", flag == 2 eig of Y"Y (snapshot method).

        #     Returns
        #     -------
        #     POD_Basis: ndarray, shape (n_x,l)
        #                 matrix containing the POD-basis vectors
        #     POD_Values: ndarray, shape (l,)
        #            vector containing the eigenvalues of Yhat (see below)

        """
        
        ### INITIALIZATION
        start_time = time()
        # set truncation tol
        tol = 1e-15
        truncate_normalized_POD_values = True
        truncate_small_POD_modes = True
        self.basissize = l
        
        if W is None:
            W = self.W_chol

        if type(D) == list and 0:   
            Dsqrt = [d.sqrt() for d in D]
        else:
            Dsqrt = np.sqrt(self.model.D)
            
        if type(Y) == list:
            if len(Y)==1:
                Y = Y[0]
            else:
                K = len(Y)
                Dsqrt = [sps.diags(Dsqrt)]*K
                Dsqrt = sps.block_diag(Dsqrt)
                Y = np.concatenate( Y, axis=1 )
        
        ### COMPUTE POD BASIS AND EIGENVALUES
        if flag == 0: 
            # singular value decomposition
            Y_hat = W@Y@Dsqrt
            l_min = min(l,min(Y_hat.shape)-1)
            if l != l_min:
                self.basissize = l_min
                print(f"Basissize dropped from {l} to {l_min} due to rank condition of snapshot matrix.")
            U, S, V = sps.linalg.svds(Y_hat, k=l_min)
            POD_values = S**2
        elif flag == 1: 
            # compute eigenvalues of YY" with size (n_x, n_x):
            Y_hat = W@Y@Dsqrt
            Y_YT = Y_hat@Y_hat.T
            POD_values, U = sps.linalg.eigsh(Y_YT, k = l, which = "LM")
        elif flag == 2: 
            # method of snapshots: eigs of Y"Y with size (n_t,n_t)
            YT_Y = Dsqrt@Y.T@W@Y@Dsqrt
            POD_values, U = sps.linalg.eigsh(YT_Y, which = "LM", k = l)
        else:
            assert 0, "wrong flag input ..."
            
        # sort eigenvalues from biggest to lowest
        U = np.fliplr(U)
        POD_values = np.flipud(POD_values)
        
        # truncate w.r.t. the normalized singular values
        if truncate_normalized_POD_values:
            normalized_values = POD_values/POD_values[0]#abs(POD_values)/POD_values[0]
        else:
            normalized_values = POD_values
        indices = normalized_values > tol
        if flag == 0:
            print(f"Smallest singular value {normalized_values[-1]} and biggest {normalized_values[0]}.")
        
        # truncate small POD modes 
        if truncate_small_POD_modes:
            POD_values = POD_values[indices]
            U = U[:,indices]
            if l_min != U.shape[1]:
                self.basissize = U.shape[1]
                print(f"Basissize dropped from {l_min} to {U.shape[1]} due to truncation of small modes.")
        
        # get POD basis
        if flag in [0,1]:
            POD_Basis = linalg.solve_triangular(W, U, lower = False)
        elif flag == 2:
            POD_Basis = Y@Dsqrt@U*1/(np.sqrt(POD_values))

        # Normalize each POD basis vector
        for i in range(POD_Basis.shape[1]):
            POD_Basis[:, i] = POD_Basis[:, i] / np.linalg.norm(POD_Basis[:, i]) #self.model.eval_L2H_norm(POD_Basis[:, i],space_norm=self.space_norm,spatial_only=True)
    
        # SAVE TO MODEL AND PRINT
        self.POD_Basis = POD_Basis
        self.POD_values = POD_values
        self.POD_values_normalized = normalized_values[indices]
        self.Singular_values = np.sqrt(POD_values)

        end_time = time()
        self.offline_time_basisconstruction = end_time - start_time

        print("\nPOD eigenvalues (lambda_i):")
        for i, val in enumerate(POD_values):
            print(f"  mode {i+1}: {val}")

        print(f"\nPOD basis shape: {POD_Basis.shape}")
        for i in range(POD_Basis.shape[1]):
            norm = self.model.eval_L2H_norm(POD_Basis[:, i], spatial_only=True)
            print(f"Norm of POD basis vector {i+1}: {norm}")
        
        return POD_Basis, POD_values
    
    def project(self, U, Y_d, U_d, U_0, V=None):
        # init
        start_time = time()
        model = self.model
        if V is None: #then do Galerkin projection
            V = U

        # create projected pde
        model.A = U.T@(model.A.dot(V))
        model.A_adj = U.T@(model.A_adj.dot(V))
        model.M = U.T@(model.M.dot(V))
        model.B = U.T@model.B@ U
        model.F = U.T @ model.F
        model.chi = U.T @ (self.model.chi[:, None] * U)
        model.solve = factorized( model.M + model.delta_t * model.A )
        model.solve_adj = factorized(model.M + model.delta_t * model.A_adj)
        model.dof = model.M.shape[0]
        model.state_dof = model.dof
        model.control_dof = model.dof
        model.y0 =  U.T@model.y0
        model.is_reduced = True
        model.update_state_products()

        Y_d_proj = U.T @ Y_d
        U_d_proj = U.T @ U_d
        U_0_proj = U.T @ U_0
        
        end_time = time()
        self.offline_time_projection = end_time - start_time
        print(f"Projected model: state_dof={model.state_dof}, control_dof={model.control_dof}")

        return Y_d_proj, U_d_proj, U_0_proj
    
    def plot_pod_values(self,path,otherpodvalues=None,otherpodvalues_normalized=None,x_axis="POD rank",fsize=14):
        l = len(self.POD_values)
        x_values = np.arange(1,l+1)
        if otherpodvalues is None:
            # POD eigenvalue decay
            plt.figure()
            plt.semilogy(x_values,self.POD_values,marker="o")
            plt.xlabel(x_axis, fontsize = fsize)
            plt.xticks(fontsize = fsize)
            plt.yticks(fontsize = fsize)
            plt.grid(True, alpha=0.3, ls="--")
            plt.tight_layout()
            plt.savefig(path+"POD_eigenvalues_decay.png",dpi=600)
            plt.close()

            # Normalized POD eigenvalue decay
            plt.figure()
            plt.semilogy(x_values,self.POD_values_normalized,marker="o")
            plt.xlabel(x_axis, fontsize = fsize)
            plt.xticks(fontsize = fsize)
            plt.yticks(fontsize = fsize)
            plt.grid(True, alpha=0.3, ls="--")
            plt.tight_layout()
            plt.savefig(path+"POD_eigenvalues_decay_normalized.png",dpi=600)
            plt.close()

            print("\nPOD eigenvalues lambda_i:")
            for i, x in enumerate(x_values):
                y = self.POD_values[i]
                print("lambda_"+str(i)+"="+str(y))
            print("\nPOD eigenvalues lambda_i normalized w.r.t. the largest eigenvalue:")
            for i, x in enumerate(x_values):
                y = self.POD_values_normalized[i]
                print("lambda_"+str(i)+"="+str(y))
        else:
            l2 = len(otherpodvalues)
            x_values = np.arange(1,max(l,l2)+1)
            # POD eigenvalue decay
            plt.figure()
            plt.semilogy(x_values[:l],self.POD_values,"o-",label="optimal")
            plt.semilogy(x_values[:l2],otherpodvalues,"s-",label="initial")
            plt.xlabel(x_axis, fontsize = fsize)
            plt.xticks(fontsize = fsize)
            plt.yticks(fontsize = fsize)
            plt.legend(loc="best", fontsize=fsize,markerscale=2.0)
            plt.grid(True, alpha=0.3, ls="--")
            plt.tight_layout()
            plt.savefig(path+"POD_eigenvalues_decay.png",dpi=600)
            plt.close()

            # Normalized POD eigenvalue decay
            plt.figure()
            plt.semilogy(x_values[:l],self.POD_values_normalized,"o-",label="optimal")
            plt.semilogy(x_values[:l2],otherpodvalues_normalized,"s-",label="initial")
            plt.xlabel(x_axis, fontsize = fsize)
            plt.xticks(fontsize = fsize)
            plt.yticks(fontsize = fsize)
            plt.legend(loc="best", fontsize=fsize,markerscale=2.0)
            plt.grid(True, alpha=0.3, ls="--")
            plt.tight_layout()
            plt.savefig(path+"POD_eigenvalues_decay_normalized.png",dpi=600)
            plt.close()