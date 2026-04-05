# POD-Based Control Space Reduction for Convection-Diffusion-Reaction Equations
Given the time interval $(0,T)$ with $T=2$ and the unit square $\Omega=(0,1)^2$, we consider the parabolic linear-quadratic optimal control problem

$\mathrm{min}_{y\in Y,u\in U} J(y,u) = \frac{1}{2}\| y-y_d\|_Y^2+\frac{\sigma}{2}\|u\|_U^2$

subject to the convection-diffusion-reaction equation

$y_t-\kappa\Delta y+\beta\cdot\nabla y+\gamma y=bu\text{ in } (0,T)\times\Omega$,

$y=0\text{on } (0,T)\times\partial\Omega$,

$y(0,\cdot)&=y_0(\cdot)&&\text{in } \Omega$,

with constant partial differential equation (PDE) parameters $\kappa>0$, $\beta\in\mathbb{R}^2$, and $\gamma\in\mathbb{R}$. We consider the desired state $y_d=1$ and varying values for the regularization $\sigma>0$.

The optimality system has the form

$\sigma\bar{u}-b\bar{p}=0$,

$-\bar{p}_t-\kappa\Delta\bar{p}-\beta\cdot\nabla\bar{p}+\gamma\bar{p}-(y_d-\bar{y})=0$,

$\bar{y}_t-\kappa\Delta\bar{y}+\beta\cdot\nabla\bar{y}+\gamma\bar{y}-b\bar{u}=0$.

We want to apply Proper Orthognal Decomposition (POD) as a model order reduction method to reduce the state space $Y$ and the control space $U$.

More details: coming.