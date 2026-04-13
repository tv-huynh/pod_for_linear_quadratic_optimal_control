# POD-Based Control Space Reduction for Convection-Diffusion-Reaction Equations
In this repository, we provide the code for the numerical experiments of the mathematics master thesis "POD-Based Control Space Reduction for Convection-Diffusion-Reaction Equations" by Thanh-Van Huynh that is currently being written at the University at Konstanz. The code is subject to change until the submission of the thesis.

## Setup
To run the code you need to install the python package FEniCS 2019 in your (local) environment together with SciPy, NumPy and Matplotlib. This can be done using `conda` via
```
conda create -n control_reduction -c conda-forge python=3.9 fenics=2019.1.0 numpy scipy matplotlib

```
After installing all the packages, run one of the experiments by
```
conda activate pod_for_linear_quadratic_optimal_control
python main.py
```

## Organization of the repository

The code consists of the following files:

* `main.py`: the main file for the experimens in Sections 6.3 to 6.6,
* `supplements.py`: discretizes the problem and contains the implementation of the full-order model (FOM) or reduced-order model (ROM),
* `optimization.py:` optimization via a Barzilai-Borwein gradient method,
* `reduce.py`: reduces the full-order model to obtain a reduced-order model.

## Theoretical Background
We aim to conduct a model-order reduction for a linear-quadratic optimal control problem (OCP) governed by the convection-diffusion-reaction (CDR) equation, a partial differential equation (PDE). To do so, we utilize the Proper Orthogonal Decomposition (POD) method, a data-driven technique based on so-called snapshots, which is typically used to reduce the state space, leading to a faster computation of the gradient when applying a gradient descent method. In this work, on the other hand, we also reduce the control space via POD. This idea combines the concept of the [variational discretization](https://link.springer.com/article/10.1007/s10589-005-4559-5) with the POD setting. We assess the performance of the fully reduced model for varying regularization parameter, various snapshot choices, different control domains and varying PDE parameters, and verify an a-posteriori estimate.

### Optimal Control Problem
Given the time interval $(0,T)$ with $T=2$ and the unit square $\Omega=(0,1)^2$, we consider the parabolic linear-quadratic optimal control problem

$$\mathrm{min}_{y\in Y,u\in U} J(y,u) = \frac{1}{2}\| y-y_d\|_Y^2+\frac{\sigma}{2}\|u\|_U^2$$

subject to the convection-diffusion-reaction equation

$$\begin{aligned}
y_t-\kappa\Delta y+\beta\cdot\nabla y+\gamma y &= bu && \text{in } (0,T)\times\Omega, \\
y &= 0 && \text{on } (0,T)\times\partial\Omega, \\
y(0,\cdot) &= y_0(\cdot) && \text{in } \Omega,
\end{aligned}$$

with constant partial differential equation (PDE) parameters $\kappa>0$, $\beta\in\mathbb{R}^2$, and $\gamma\in\mathbb{R}$. We consider the desired state $y_d=1$ and varying values for the regularization $\sigma>0$. For a control domain $\omega\subset\Omega$, we denote by $b(x):=\chi_{\omega}(x)$ the characteristic function, which is 1 for values in $\omega$ and 0 for values in $\Omega\setminus\omega$.

### Optimality System
$(\bar{y},\bar{u})$ is a solution to the optimal control problem if and only if $(\bar{y},\bar{u})$ satisfy together with the adjoint variable $\bar{p}$ the first-order optimality system

$$\begin{aligned}
\sigma\bar{u}-b\bar{p} &= 0, \\
-\bar{p}_t-\kappa\Delta\bar{p}-\beta\cdot\nabla\bar{p}+\gamma\bar{p}-(y_d-\bar{y}) &= 0, \\
\bar{y}_t-\kappa\Delta\bar{y}+\beta\cdot\nabla\bar{y}+\gamma\bar{y}-b\bar{u} &= 0.
\end{aligned}$$

### Proper Orthogonal Decomposition
We want to apply POD to reduce the state space (where $y$ and $p$ live) as well as the control space (where $u$ lives). Given data in the form of snapshots, the POD algorithm finds a POD basis of rank $\ell$ (which is much smaller than the degrees of freedom of the full-order model) which captures the maximum of variation/energy in the snapshot data. This POD basis spans the reduced state and control spaces. We consider an approach based on singular decomposition, but also implement one based on the eigenvalue decomposition which is more efficient numerically for problems of higher dimensions. 

There are different choices for the snapshot set $S$: we consider initial snapshots $S=\{y(u_0),p(u_0)\}$ with $u_0$ being the initial control value and optimal snapshots $S=\{y(\bar{u}),p(\bar{u})\}$ with $\bar{u}$ being the optimal control of the full-order model.

### Control Space Reduction
We recall the optimality system and write the sensitivity equation as

$$\bar{u}=\frac{1}{\sigma}b\bar{p}.$$

Inferring a POD approximation leads to an approximation of the adjoint variable, denoted by $p^{\ell}$, and the corresponding optimality condition becomes

$$\bar{u}^{\ell}=\frac{1}{\sigma}b\bar{p}^{\ell}=\frac{1}{\sigma}\sum_{i=1}^{\ell} \mathbf{p}_i\chi_{\omega}\bar{\psi}_i$$

for coefficients $\mathbf{p}_i\in\mathbb{R}$ from the representation of $\bar{p}^{\ell}$ w.r.t. the POD basis $\{\bar{\psi}_i\}$, i.e., the approximation of the control space is inherited from the approximation of the optimal adjoint through the optimality condition. [In fact](https://arxiv.org/abs/2510.14479 "as shown by Kartmann and Volkwein"), the induced control space reduction is quite natural, in the sense that both the state-reduced and the fully reduced OCP admit the same optimizer.

### Error Estimates
Let us consider an FE solution $\bar{u}$ of the full-order model and a POD solution $\bar{u}^{\ell}$ of the reduced-order model. We want to estimate $\|\bar{u}-\bar{u}^{\ell}\|_U$.
* A-priori: TODO
* A-posteriori: The accuracy of any given solution $\bar{u}^{\ell}$ approximating the exact solution $\bar{u}$ is given by the a-posteriori error estimate
$$\|\bar{u}-\bar{u}^{\ell}\|_U\leq\frac{1}{\sigma}\|\nabla\hat{J}(\bar{u}^{\ell})\|_U=\frac{1}{\sigma}\|\sigma\bar{u}^{\ell}-b\bar{p}^{\ell}\|_U.$$