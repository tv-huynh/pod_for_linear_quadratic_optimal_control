"""
ES6: supplements
Original work by: Andrea Petrocchi (July 2023)
Modifications and additions by: Thanh-Van Huynh
"""

import os, shutil, pdb
import fenics
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, save_npz
from scipy.sparse.linalg import spsolve, factorized
import matplotlib.pyplot as plt
import time


#%% analytical problem
class analytical_problem():
    def __init__(self):
        self.dir_boundary = lambda x,on_boundary: on_boundary
        self.c_u = fenics.Constant(1.0)
        self.true_solution = None
    @property
    def SE(self):
        return fenics.Point(self.x1a,self.x2a)
    @SE.setter
    def SE(self,new_Point):
        self.x1a=new_Point[0]; self.x2a=new_Point[1]
    @property
    def NW(self):
        return fenics.Point(self.x1b,self.x2b)
    @NW.setter
    def NW(self,new_Point):
        self.x1b=new_Point[0]; self.x2b=new_Point[1]

#%% parabolic problem
class parabolic_model():
    def __init__( self, p ):
        self.p = p
        self.control_dof = None
        self.state_dof = None
    
    def build_problem( self ):
        self.build_mesh()
        self.build_timegrid()
        self.define_function_space()
        self.create_FE_matrices()
     
    def build_timegrid(self):
        self.K = self.p.K
        self.delta_t = (self.p.T-self.p.t0)/(self.K-1)
        self.t_v = np.linspace( self.p.t0, self.p.T, num=self.K )
        self.D = self.delta_t * np.ones(self.K) 
        self.D[0] = self.D[-1] = self.delta_t/2
        
    def build_mesh( self ):
        i = self.p.h / np.sqrt(2)
        l1 = self.p.x1b - self.p.x1a
        l2 = self.p.x2b - self.p.x2a
        Nx = int(l1/i) + 1; self.Nx = Nx
        Ny = int(l2/i) + 1; self.Ny = Ny
        self.mesh = fenics.RectangleMesh( self.p.SE, self.p.NW, Nx, Ny )
        self.h = self.mesh.hmax()
        #print("Check maximum cell size: {}".format(self.mesh.hmax()))
        if self.mesh.hmax() > self.p.h-1.e-20:
            breakpoint()
        # Plot mesh
        plt.figure()
        fenics.plot(self.mesh, linewidth=1.5, color="tab:blue")
        plt.title("Mesh")
        plt.savefig("plots/mesh.png")
        plt.close()
        
    def define_function_space( self ):
        self.V = fenics.FunctionSpace( self.mesh, 'CG', 1 )
        self.BC = fenics.DirichletBC( self.V, fenics.Constant(0.0), self.p.dir_boundary )        
            
    def create_FE_matrices( self ):
        y = fenics.TrialFunction(self.V)
        phi = fenics.TestFunction(self.V)
        M = fenics.assemble( y * phi * fenics.dx ) # mass matrix
        A = fenics.assemble( fenics.dot(fenics.nabla_grad(y),
                fenics.nabla_grad(phi)) * fenics.dx ) # stiffness matrix 
        B = fenics.assemble( self.p.c_u * y * phi * fenics.dx )
        self.y0 = fenics.interpolate(self.p.y0,self.V).vector().get_local()
        self.BC.apply(M)
        self.BC.apply(A)
        self.BC.apply(B)
        self.M = csc_matrix(fenics.as_backend_type(M).mat().getValuesCSR()[::-1])
        self.A = csc_matrix(fenics.as_backend_type(A).mat().getValuesCSR()[::-1])
        self.B = csc_matrix(fenics.as_backend_type(B).mat().getValuesCSR()[::-1])
        self.solve = factorized( self.M + self.delta_t * self.A )
        self.dof = self.M.shape[0]
        self.control_dof = self.dof # initially, state and control dimensions are the same
        self.state_dof = self.dof
        self.state_products = {"H1": self.A, "L2": self.M, "H10": self.M+self.A}
        self.is_reduced = False # flag to track if model is reduced

    def update_state_products(self):
        self.state_products = {"H1": self.A, "L2": self.M, "H10": self.M+self.A}
    
    def matrix_to_vector( self, V ):
        return V.T
    
    def vector_to_matrix( self, v, option="state" ):
        if option == "state":
            """Reshape state vector using state dimension"""
            return v.reshape(self.state_dof,self.K)
        elif option == "control":
            """Reshape control vector using control dimension"""
            return v.reshape(self.control_dof, self.K)
    
    def solve_state( self, U ):
        y = self.y0.copy(); Y = y.copy().reshape(-1,1)
        for k in range( 1, self.K ):
            b = self.M.dot(y) + self.B.dot(U[:,k])
            b = self.apply_BC_to_vector(b)
            y = self.solve( b )
            Y = np.concatenate((Y,y.reshape(-1,1)), axis=1)
        return Y
    
    def solve_adjoint( self, Z ):
        # Z = Yd - Y
        F = self.delta_t * self.M @ Z
        p = np.zeros_like(Z[:,-1]); P = p.copy().reshape(-1,1)
        for k in range( self.K-2, -1, -1 ):
            b = self.M.dot(p) + F[:,k]
            b = self.apply_BC_to_vector(b)
            p = self.solve(b)
            P = np.concatenate( (p.reshape(-1,1),P), axis=1 )
        return P
    
    def eval_L2H_prod( self, v, w, space_norm="L2"):
        # Ensure v and w are matrices
        if len(v.shape) == 1:
            if space_norm == "control":
                v = self.vector_to_matrix(v, option="control")
            else:
                v = self.vector_to_matrix(v, option="state")
        if len(w.shape) == 1:
            if space_norm == "control":
                w = self.vector_to_matrix(w, option="control")
            else:
                w = self.vector_to_matrix(w, option="state")
        
        # Use mass matrix for ALL norms (including control)
        if space_norm == "control" or space_norm == "L2":
            S = self.M
        elif space_norm == "H10":
            S = self.A
        elif space_norm == "H1":
            S = self.M + self.A
        else:
            raise ValueError("Unknown space_norm")

        return np.vdot(self.D, np.diag(v.T.dot(S.dot(w))))

    
    def eval_L2H_norm( self, v ,space_norm="L2"):
        return np.sqrt( self.eval_L2H_prod(v,v,space_norm) )

    def apply_BC_to_matrix( self, M ):
        if self.is_reduced: # don't apply BC to reduced matrices
            return M
        bc = self.BC.get_boundary_values()
        D = csc_matrix(np.diag([bc[k] if k in bc else 1.0 for k in range(M.shape[0])]))
        M = D.dot(M.dot(D))
        return M
    
    def apply_BC_to_vector( self, b ):
        if self.is_reduced: # don't apply BC to reduced vectors
            return b
        bc = self.BC.get_boundary_values()
        b = [ bc[k] if k in bc else b[k] for k in range(b.size) ]
        return np.array(b)

    def get_snapshots(self, u, Y_d):
        U = self.vector_to_matrix(u,option="control")
        Y = self.solve_state(U)
        P = self.solve_adjoint(Y_d-Y)
        return Y, P
        

    #%% PLOTS AND FIGURES
    def format_folder( self, path ):
        if os.path.isdir(path):
            shutil.rmtree(path) # if folder exists, delete it
        os.mkdir(path) # create flder
        
    def fenics_plot_solution( self, y, title='' ):
        yf = fenics.Function(self.V)
        yf.vector()[:] = y
        fenics.plot(yf, title=title)
        plt.figure()
    
    def save_vtk( self, Y, name ):
        vtkfile = fenics.File(name)
        for k in range(Y.shape[1]):
            y_k = fenics.Function(self.V)
            y_k.vector()[:] = Y[:,k].copy()
            t_k = k * self.delta_t
            vtkfile << (y_k,t_k)

    def plot_3d( self, y, title=None, save_png=False, path=None, dpi='figure' ):
        dims = ( self.Ny+1 , self.Nx+1 )
        X = np.reshape( self.mesh.coordinates()[:,0], dims )
        Y = np.reshape( self.mesh.coordinates()[:,1], dims )
        Z = np.reshape( y[fenics.vertex_to_dof_map(self.V)], dims )
        fig = plt.figure()
        ax = fig.add_subplot( projection='3d' )
        if title is not None:
            ax.set_title(title)
        surf = ax.plot_surface( X, Y, Z, cmap=plt.cm.coolwarm )
        fig.colorbar(surf, shrink=0.75, aspect=10)
        # fig.colorbar(surf)
        plt.tight_layout()
        if save_png:
            plt.savefig(path,dpi=600)
        plt.close(fig)

