"""
supplements.py
@author: Thanh-Van Huynh
"""

import os, shutil
import fenics
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import factorized
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation


#%% analytical problem
class analytical_problem():
    def __init__(self):
        self.dir_boundary = lambda x,on_boundary: on_boundary
        self.true_solution = None
        self.kappa = fenics.Constant(1.0)
        self.beta_x1 = fenics.Constant(1.0)
        self.beta_x2 = fenics.Constant(0.0)
        self.beta = fenics.as_vector((self.beta_x1, self.beta_x2))
        self.gamma = fenics.Constant(0.0)
        self.f = 0
        self.omega = [((0.0, 1.0), (0.0, 1.0))]  # list of ((x0,x1),(y0,y1)) tuples, default:full domain

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
        self.is_reduced = False
    
    def build_problem( self ):
        self.build_mesh()
        self.build_timegrid()
        self.define_function_space()
        self.create_FE_matrices()
        full_domain = [((0.0, 1.0), (0.0, 1.0))]
            
    def build_timegrid(self):
        print("Time steps K="+str(self.p.K))
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
        print("Mesh size h="+str(self.p.h))
        print("Degrees of freedom N="+str(Nx*Ny))
        if self.mesh.hmax() > self.p.h-1.e-20:
            breakpoint()
        
    def define_function_space( self ):
        self.V = fenics.FunctionSpace( self.mesh, "CG", 1 )
        self.BC = fenics.DirichletBC( self.V, fenics.Constant(0.0), self.p.dir_boundary )        
            
    def create_FE_matrices( self ):
        kappa = self.p.kappa
        beta = self.p.beta
        gamma = self.p.gamma
        print(f"Diffusion coefficient kappa = {float(kappa)}")
        print(f"Convection coefficient beta = {(float(self.p.beta_x1),float(self.p.beta_x2))}")
        print(f"Reaction coefficient gamma = {float(gamma)}")
        y = fenics.TrialFunction(self.V)
        phi = fenics.TestFunction(self.V); self.phi = phi
        M = fenics.assemble( y * phi * fenics.dx ) # mass matrix
        A = fenics.assemble( 
            (
                kappa * (fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(phi)))
                + fenics.dot(beta, fenics.nabla_grad(y)) * phi
                + gamma * y * phi
            ) * fenics.dx ) # stiffness matrix 
        A_adj = fenics.assemble( 
            (
                kappa * (fenics.dot(fenics.nabla_grad(y),fenics.nabla_grad(phi)))
                - fenics.dot(beta, fenics.nabla_grad(y)) * phi
                + gamma * y * phi
            ) * fenics.dx ) # adjoint stiffness matrix 
        conditions = " || ".join([
            "((x[0] >= {x0}) && (x[0] <= {x1}) && (x[1] >= {y0}) && (x[1] <= {y1}))".format(
                x0=r[0][0], x1=r[0][1], y0=r[1][0], y1=r[1][1])
            for r in self.p.omega
        ])
        chi_omega = fenics.Expression(f"({conditions}) ? 1.0 : 0.0", degree=1)
        chi_dof = fenics.interpolate(chi_omega, self.V).vector().get_local()
        self.chi = chi_dof  # 1D array of χ_ω nodal values
        B = fenics.assemble( chi_omega * y * phi * fenics.dx )

        self.y0 = fenics.interpolate(self.p.y0,self.V).vector().get_local()
        if isinstance(self.p.f, fenics.Expression) or isinstance(self.p.f, fenics.Constant):
            self.f_expr = self.p.f
        else:
            self.f_expr = None
        self.BC.apply(M)
        self.BC.apply(A)
        self.BC.apply(A_adj)
        self.B = csr_matrix(fenics.as_backend_type(B).mat().getValuesCSR()[::-1])
        self.M = csr_matrix(fenics.as_backend_type(M).mat().getValuesCSR()[::-1])
        self.A = csr_matrix(fenics.as_backend_type(A).mat().getValuesCSR()[::-1])
        self.A_adj = csr_matrix(fenics.as_backend_type(A_adj).mat().getValuesCSR()[::-1])
        self.B = csr_matrix(fenics.as_backend_type(B).mat().getValuesCSR()[::-1])
        self.solve = factorized( self.M + self.delta_t * self.A )
        self.solve_adj = factorized( self.M + self.delta_t * self.A_adj )
        self.dof = self.M.shape[0]
        self.control_dof = self.dof
        self.state_dof = self.dof
        self.state_products = {"H1": self.A, "L2": self.M, "H10": self.M+self.A}
        self.F = self.get_F_matrix()

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
    
    def get_F_matrix(self):
        F = np.zeros((self.dof, self.K))
        if self.f_expr is None:
            return F
        for k in range(self.K):
            if hasattr(self.f_expr, "t"):
                self.f_expr.t = float(self.t_v[k])
            f_vec = fenics.assemble(self.f_expr * self.phi * fenics.dx).get_local()
            F[:, k] = f_vec
        return F

    def solve_state( self, U ):
        y = self.y0.copy(); Y = y.copy().reshape(-1,1)
        for k in range( 1, self.K ):
            b = self.M.dot(y) + self.B.dot(U[:,k])
            
            if self.is_reduced:
                # use projected forcing
                if hasattr(self, "F"):
                    b += self.delta_t * self.F[:, k]
            else:
                # FOM: assemble f from Expression
                if self.f_expr is not None:
                    b += self.delta_t * self.F[:, k]

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
            p = self.solve_adj(b)
            P = np.concatenate( (p.reshape(-1,1),P), axis=1 )
        return P
    
    def eval_L2H_prod( self, v, w, space_norm="L2",spatial_only=False):
        # Set gramian
        if space_norm == "control" or space_norm == "L2":
            S = self.M
        elif space_norm == "H10":
            S = self.A
        elif space_norm == "H1":
            S = self.M + self.A
        else:
            raise ValueError("Unknown space_norm")
        
        # If spatial_only, treat inputs as spatial vectors only (no reshaping)
        if spatial_only:
            return np.vdot(v, S.dot(w))
        
        # Otherwise treat as space-time vectors and reshape
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

        return np.vdot(self.D, np.diag(v.T.dot(S.dot(w))))

    def eval_L2H_norm( self, v ,space_norm="L2",spatial_only=False):
        return np.sqrt( self.eval_L2H_prod(v,v,space_norm,spatial_only) )

    def apply_BC_to_matrix( self, M ):
        if self.is_reduced: # don"t apply BC to reduced matrices
            return M
        bc = self.BC.get_boundary_values()
        D = csr_matrix(np.diag([bc[k] if k in bc else 1.0 for k in range(M.shape[0])]))
        M = D.dot(M.dot(D))
        return M
    
    def apply_BC_to_vector( self, b ):
        if self.is_reduced: # don"t apply BC to reduced vectors
            return b
        bc = self.BC.get_boundary_values()
        b = [ bc[k] if k in bc else b[k] for k in range(b.size) ]
        return np.array(b)

    def get_snapshots(self, u, Y_d):
        U = self.vector_to_matrix(u,option="control")
        Y = self.solve_state(U)
        P = self.solve_adjoint(Y_d-Y)
        return [Y, P]
        

    #%% PLOTS AND FIGURES
    def format_folder( self, path ):
        if os.path.isdir(path):
            shutil.rmtree(path) # if folder exists, delete it
        os.mkdir(path) # create flder
        
    def save_vtk( self, Y, name ):
        vtkfile = fenics.File(name)
        for k in range(Y.shape[1]):
            y_k = fenics.Function(self.V)
            y_k.vector()[:] = Y[:,k].copy()
            t_k = k * self.delta_t
            vtkfile << (y_k,t_k)

    def plot_3d(self, y, title=None, save_png=False, path=None, colorbar=False, dpi="figure"):
        dims = ( self.Ny+1 , self.Nx+1 )
        X = np.reshape( self.mesh.coordinates()[:,0], dims )
        Y = np.reshape( self.mesh.coordinates()[:,1], dims )
        Z = np.reshape( y[fenics.vertex_to_dof_map(self.V)], dims )
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        surf = ax.plot_surface( X, Y, Z, cmap=plt.cm.coolwarm )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlim(0, Z.max())
        if title is not None:
            ax.set_title(title)
        if colorbar:
            fig.colorbar(surf, shrink=0.75, aspect=10)
        plt.tight_layout()
        if save_png:
            plt.savefig(path,dpi=600,bbox_inches="tight",pad_inches=0.05)
        plt.close(fig)

    def plot_beta(self, title=None, save_png=True, path="plots/beta",fsize=18):
        V_vec = fenics.VectorFunctionSpace(self.mesh, "CG", 1)
        
        # Interpolate components separately
        beta_x1_fun = fenics.Function(self.V)
        beta_x1_fun.interpolate(self.p.beta_x1)
        beta_x2_fun = fenics.Function(self.V)
        beta_x2_fun.interpolate(self.p.beta_x2)
        
        # Get vertex values as (n_verts, 2)
        vals_x1 = beta_x1_fun.compute_vertex_values(self.mesh)
        vals_x2 = beta_x2_fun.compute_vertex_values(self.mesh)
        vals = np.stack([vals_x1, vals_x2], axis=1)
        
        # Subsample for clarity
        n_arrows = 400
        step = max(1, len(vals) // n_arrows)
        pts = self.mesh.coordinates()[::step]
        vecs = vals[::step]
        
        # Scale lengths: if ||β||=1 → original length
        mags = np.linalg.norm(vecs, axis=1)
        vecs_scaled = vecs / np.max(mags)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.quiver(pts[:, 0], pts[:, 1], 
                vecs_scaled[:, 0], vecs_scaled[:, 1],
                color="tab:blue",  # Pure coolest blue
                scale=20, width=0.004, alpha=0.8)
        plt.axis("equal")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.margins(0)
        plt.xticks(fontsize = fsize)
        plt.yticks(fontsize = fsize)
        plt.xlabel("$x_1$", fontsize=fsize)
        plt.ylabel("$x_2$", fontsize=fsize)
        if title != None:
            plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_png:
            plt.savefig(f"{path}.png", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_mesh(self, fsize=18, save_png=True, path="plots/mesh"):
        plt.figure()
        plt.figure(figsize=(8, 8))
        fenics.plot(self.mesh, linewidth=1.5, color="tab:blue")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.margins(0)
        plt.xticks(fontsize = fsize)
        plt.yticks(fontsize = fsize)
        plt.xlabel("$x_1$", fontsize=fsize)
        plt.ylabel("$x_2$", fontsize=fsize)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_png:
            plt.savefig(f"{path}.png", dpi=600, bbox_inches="tight")
        plt.close()
    
    def plot_error_vs_x(self, x, err_u, err_y, err_p, err_u_2=None, err_y_2=None, err_p_2=None, firstgroup="abs ", secondgroup="rel ", axis="normal", x_axis=None, title=None, save_png=True, path=None, fsize=18):
        """Plot err_u/y/p vs x (with normal or semilogy or loglog axis)."""
        plt.figure(figsize=(8, 6))
        if axis=="normal":
            if err_u_2 is None:
                plt.plot(x, err_u, "o-", label="$err_u$", linewidth=2, markersize=6)
                plt.plot(x, err_y, "s-", label="$err_y$", linewidth=2, markersize=6)
                plt.plot(x, err_p, "^-", label="$err_p$", linewidth=2, markersize=6)
            else:
                plt.plot(x[:len(err_u)], err_u, "o-", label=firstgroup+"$err_u$", linewidth=2, markersize=6)
                plt.plot(x[:len(err_y)], err_y, "s-", label=firstgroup+"$err_y$", linewidth=2, markersize=6)
                plt.plot(x[:len(err_p)], err_p, "^-", label=firstgroup+"$err_p$", linewidth=2, markersize=6)
                plt.plot(x[:len(err_u_2)], err_u_2, "o--", color="tab:blue", label=secondgroup+"$err_u$", linewidth=2, markersize=6)
                plt.plot(x[:len(err_y_2)], err_y_2, "s--", color="tab:orange", label=secondgroup+"$err_y$", linewidth=2, markersize=6)
                plt.plot(x[:len(err_p_2)], err_p_2, "^--", color="tab:green", label=secondgroup+"$err_p$", linewidth=2, markersize=6)
        elif axis=="semilogy":
            if err_u_2 is None:
                plt.semilogy(x, err_u, "o-", label="$err_u$", linewidth=2, markersize=6)
                plt.semilogy(x, err_y, "s-", label="$err_y$", linewidth=2, markersize=6)
                plt.semilogy(x, err_p, "^-", label="$err_p$", linewidth=2, markersize=6)
            else:
                plt.semilogy(x[:len(err_u)], err_u, "o-", label=firstgroup+"$err_u$", linewidth=2, markersize=6)
                plt.semilogy(x[:len(err_y)], err_y, "s-", label=firstgroup+"$err_y$", linewidth=2, markersize=6)
                plt.semilogy(x[:len(err_p)], err_p, "^-", label=firstgroup+"$err_p$", linewidth=2, markersize=6)
                plt.semilogy(x[:len(err_u_2)], err_u_2, "o--", color="tab:blue", label=secondgroup+"$err_u$", linewidth=2, markersize=6)
                plt.semilogy(x[:len(err_y_2)], err_y_2, "s--", color="tab:orange", label=secondgroup+"$err_y$", linewidth=2, markersize=6)
                plt.semilogy(x[:len(err_p_2)], err_p_2, "^--", color="tab:green", label=secondgroup+"$err_p$", linewidth=2, markersize=6)
        elif axis=="loglog":
            if err_u_2 is None:
                plt.loglog(x, err_u, "o-", label="$err_u$", linewidth=2, markersize=6)
                plt.loglog(x, err_y, "s-", label="$err_y$", linewidth=2, markersize=6)
                plt.loglog(x, err_p, "^-", label="$err_p$", linewidth=2, markersize=6)
            else:
                plt.loglog(x[:len(err_u)], err_u, "o-", label=firstgroup+"$err_u$", linewidth=2, markersize=6)
                plt.loglog(x[:len(err_y)], err_y, "s-", label=firstgroup+"$err_y$", linewidth=2, markersize=6)
                plt.loglog(x[:len(err_p)], err_p, "^-", label=firstgroup+"$err_p$", linewidth=2, markersize=6)
                plt.loglog(x[:len(err_u_2)], err_u_2, "o--", color="tab:blue", label=secondgroup+"$err_u$", linewidth=2, markersize=6)
                plt.loglog(x[:len(err_y_2)], err_y_2, "s--", color="tab:orange", label=secondgroup+"$err_y$", linewidth=2, markersize=6)
                plt.loglog(x[:len(err_p_2)], err_p_2, "^--", color="tab:green", label=secondgroup+"$err_p$", linewidth=2, markersize=6)
        if x_axis is not None:
            plt.xlabel(x_axis, fontsize = fsize)
        if title is not None:
            plt.title(title)
        plt.xticks(fontsize = fsize)
        plt.yticks(fontsize = fsize)
        plt.legend(loc="best", fontsize=fsize,markerscale=2.0)
        if err_u_2 is None:
            plt.legend(loc="best", fontsize=fsize,markerscale=2.0)
        else:
            plt.legend(loc="best", ncol=2, fontsize=fsize,markerscale=2.0)
        plt.grid(True, alpha=0.3, ls="--")
        plt.tight_layout()
        if save_png:
            plt.savefig(f"{path}.png", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_contour_clean(self, U_slice, path="contour_clean.png"):
        """True top-view contour matching plot_3d."""
        
        fig = plt.figure(figsize=(14,4), dpi=600)
    
        # Filled contours (top-view of 3D surface)
        dims = ( self.Ny+1 , self.Nx+1 )
        X = np.reshape( self.mesh.coordinates()[:,0], dims )
        Y = np.reshape( self.mesh.coordinates()[:,1], dims )
        Z = np.reshape( U_slice[fenics.vertex_to_dof_map(self.V)], dims )
        plt.contourf(X, Y, Z, levels=30, cmap='RdYlBu_r', extend='both')
        
        # Pure image (no axes/colorbar/whitespace)
        plt.axis('off')
        plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
        
        plt.savefig(path+"contour.png", bbox_inches='tight', pad_inches=0, dpi=600, 
                    facecolor='none', edgecolor='none', transparent=True)
        plt.close()


    def plot_f_at_index(self, k, title=None, save_png=False, path=None):
        if self.f_expr is None:
            print("No forcing term f defined.")
            return
        t_k = float(self.t_v[k])          # time at index k
        if hasattr(self.f_expr, "t"):
            self.f_expr.t = t_k
        f_fun = fenics.interpolate(self.f_expr, self.V)
        vals = f_fun.vector().get_local()
        self.plot_3d(vals, title=title, save_png=save_png, path=path)
    
    def save_mp4(self, Y, path, title=None, fps=10, dpi=150):
        dims = (self.Ny + 1, self.Nx + 1)
        X = np.reshape(self.mesh.coordinates()[:, 0], dims)
        Ym = np.reshape(self.mesh.coordinates()[:, 1], dims)
        
        vmin, vmax = Y[:, 1:-1].min(), Y[:, 1:-1].max()  # fix colorscale across all frames
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        order = "FOM"
        if self.is_reduced:
            order = "ROM"
        
        def update(k):
            ax.cla()
            Z = np.reshape(Y[fenics.vertex_to_dof_map(self.V), k], dims)
            ax.plot_surface(X, Ym, Z, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlim(vmin, vmax)
            if title:
                ax.set_title(order+" "+title+f" at t={k * self.delta_t:.2f}")
        
        ani = animation.FuncAnimation(fig, update, frames=range(1, Y.shape[1]-1), interval=1000//fps)
        ani.save(path, writer="ffmpeg", fps=fps, dpi=dpi)
        plt.close(fig)