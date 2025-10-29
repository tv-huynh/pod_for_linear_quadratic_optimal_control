"""
reduce.py
Original work by: Michael Kartmann (https://github.com/michikartmann/pod_for_linear_quadratic_optimal_control)
Modifications and additions by: Thanh-Van Huynh
"""
import scipy.sparse as sps
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized
from time import perf_counter

class pod():
    def __init__(self, model, space_product="L2"):
        self.model = model

        if space_product == "L2":
            S = model.M
        elif space_product == "H10":
            S = model.A
        elif space_product == "H1":
            S = model.M + model.A
            
        start_time = perf_counter()
        # Handle both sparse and dense matrices
        if hasattr(S, 'todense'):
            self.W_chol = linalg.cholesky(S.todense())
        else:
            self.W_chol = linalg.cholesky(S)
        end_time = perf_counter()
        print(f'Cholesky of space product done in {end_time-start_time}')


    def pod_basis(self, Y, l, W = None, D = None, flag = 0, plot = False, energy_tolerance = None):
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
        #         (if flag==0 svd, flag == 1 eig of YY', flag == 2 eig of Y'Y (snapshot method).

        #     Returns
        #     -------
        #     POD_Basis: ndarray, shape (n_x,l)
        #                 matrix containing the POD-basis vectors
        #     POD_Values: ndarray, shape (l,)
        #            vector containing the eigenvalues of Yhat (see below)

        """
        
        ### INITIALIZATION
        # set truncation tol
        tol = 1e-15
        truncate_normalized_POD_values = True
        self.basissize = l
        
        if W is None: 
            # TODO set it to euclidian product
            pass 
        if D is None:
            pass
        
        # TODO do checks that the length of snapshots can vary
        if type(D) == list and 0:   
            Dsqrt = [d.sqrt() for d in D]
        else:
            Dsqrt = np.sqrt(self.model.D)
            
        if type(Y) == list:
                
                if len(Y)==1:
                    Y = Y[0]
                else:
                    # at the moment all snapshots need to have the same length
                    K = len(Y)
                    Dsqrt = [sps.diags(Dsqrt)]*K
                    Dsqrt = sps.block_diag(Dsqrt)
                    nx, nt = Y[0].shape
                    Y = np.concatenate( Y, axis=1 )      
        elif isinstance(Y, np.ndarray):
                nx, nt = Y.shape
        
        ### COMPUTE POD BASIS
        if flag == 0:
        # SVD
        
            # scale matrix
            Y_hat = self.W_chol@Y@Dsqrt
            l_min = min(l,min(Y_hat.shape)-1)
            if l != l_min:
                self.basissize = l_min
                print(f'Basissize dropped from {l} to {l_min} due to rank condition of snapshot matrix.')
            
            # perform svd
            U, S, V = sps.linalg.svds(Y_hat, k=l_min)
            
            # get pod values
            POD_values = S**2
    
            # sort from biggest to lowest
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            # truncate w.r.t. the normalized singular values
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]#abs(POD_values)/POD_values[0]
            else:
                normalized_values = POD_values
            print(f'Smallest singular value {normalized_values[-1]} and biggest {normalized_values[0]}.')
            indices = normalized_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            if l_min != U.shape[1]:
                self.basissize = U.shape[1]
                print(f'Basissize dropped from {l_min} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            if 1:
                POD_Basis = linalg.solve_triangular(self.W_chol, U, lower = False)
            else:
                POD_Basis = U

            # Normalize each POD basis vector
            for i in range(POD_Basis.shape[1]):
                POD_Basis[:, i] = POD_Basis[:, i] / np.linalg.norm(POD_Basis[:, i])
        
        elif flag == 1: 
            # Compute eigenvalues of YY' with size (n_x, n_x):
           
            # scale matrix
            Y_hat = self.W_chol@Y@Dsqrt
            Y_YT = Y_hat@Y_hat.T
            
            # compute eigenvalues
            POD_values, U = sps.linalg.eigsh(Y_YT, k = l, which = 'LM')
            
            # sort it from the biggest to the lowest
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            # truncate w.r.t. the normalized singular values
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]
            else:
                normalized_values = POD_values   
            indices = POD_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            if l_min != U.shape[1]:
                self.basissize = U.shape[1]
                print(f'Basissize dropped from {l} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            if 1:
                POD_Basis = linalg.solve_triangular(self.W_chol, U, lower = False)
            else:
                POD_Basis = U
            
            # Normalize each POD basis vector
            for i in range(POD_Basis.shape[1]):
                POD_Basis[:, i] = POD_Basis[:, i] / np.linalg.norm(POD_Basis[:, i])

        elif flag == 2: 
            # Method of snapshots: eigs of Y'Y with size (n_t,n_t)
           
            YT_Y = Dsqrt@Y.T@W@Y@Dsqrt

            if 1:
                POD_values, U = sps.linalg.eigsh(YT_Y, which = 'LM', k = l)
            else:
                pass
            
            # sort the computed eigenvalues and eigenvectors from biggest to lowest
            U = np.fliplr(U)
            POD_values = np.flipud(POD_values)
            
            # truncate w.r.t. the normalized singular values
            if truncate_normalized_POD_values:
                normalized_values = POD_values/POD_values[0]#abs(POD_values)/POD_values[0]
            else:
                normalized_values = POD_values
            indices = normalized_values > tol
            POD_values = POD_values[indices]
            U = U[:,indices]
            if l_min != U.shape[1]:
                self.basissize = U.shape[1]
                print(f'Basissize dropped from {l} to {U.shape[1]} due to truncation of small modes.')
            
            # get POD basis
            POD_Basis = Y@Dsqrt@U*1/(np.sqrt(POD_values))

            # Normalize each POD basis vector
            for i in range(POD_Basis.shape[1]):
                POD_Basis[:, i] = POD_Basis[:, i] / np.linalg.norm(POD_Basis[:, i])
            
        else:
            assert 0, 'wrong flag input ...'
            
        self.POD_Basis = POD_Basis
        self.POD_values = POD_values
        self.Singular_values = np.sqrt(POD_values)
        
        return POD_Basis, POD_values
    
    def plot_pod_values(self,path):
        l = len(self.POD_values)
        x_values = np.arange(1,l+1)
        plt.figure()
        # plt.title("POD Eigenvalues decay")
        plt.plot(x_values,self.POD_values,marker="o")
        print("\nPOD eigenvalues lambda_i:")

        for i, x in enumerate(x_values):
            y = self.POD_values[i]
            print("lambda_"+str(i)+"="+str(y))

        plt.ylabel(r"$\lambda_i$")
        plt.xticks(x_values)
        plt.tight_layout()
        plt.savefig(path+"POD_eigenvalues_decay.png",dpi=600)
        plt.close()

    def plot_error(self, error_list, path):
        l = len(error_list)
        x_values = np.arange(1, l+1)
        plt.figure()
        plt.plot(x_values, error_list, marker="o")
        print("\nControl error || u_POD - u_FE || depending on l:")

        for i, x in enumerate(x_values):
            y = error_list[i]
            print("l="+str(i)+", err="+str(y))
        
        plt.xticks(x_values)
        plt.xlabel("number of snapshots $\ell$")
        plt.ylabel(r"$||\bar{u}^{\ell}-\bar{u}||$")
        plt.tight_layout()
        plt.savefig(path + "POD_error.png", dpi=600)
        plt.close()


    def project(self, U, Y_d, U_d, U_0, V=None):
        # init
        model = self.model
        if V is None: #then do Galerkin projection
            V = U

        # Store original matrices and POD basis
        model.M_FOM = model.M.copy()
        model.A_FOM = model.A.copy()
        model.B_FOM = model.B.copy()
        model.POD_Basis = U

        # create projected pde
        model.A = U.T@(model.A.dot(V))
        model.M = U.T@(model.M.dot(V))
        model.B = U.T@model.B@ U
        model.solve = factorized( model.M + model.delta_t * model.A )
        model.dof = model.M.shape[0]
        model.state_dof = model.dof
        model.control_dof = model.dof
        model.y0 =  U.T@model.y0
        model.is_reduced = True
        model.update_state_products()

        Y_d_proj = U.T @ Y_d # project Y_d into reduced space
        U_d_proj = U.T @ U_d # project U_d into reduced space
        U_0_proj = U.T @ U_0 # project U_0 into reduced space
        
        print(f"Projected model: state_dof={model.state_dof}, control_dof={model.control_dof}")

        return Y_d_proj, U_d_proj, U_0_proj