from sympy import Matrix, diff, simplify, symbols
from sympy import factor, lambdify, sqrt, Function, Eq, S


def laplace_beltrami(ginv, sqrt_detg, x, p):
    Del = 0
    d = x.shape[0]
    for j in range(d):
        for i in range(d):
            Del += diff(sqrt_detg * ginv[i, j] * diff(p, x[i]), x[j])
    Del = Del / sqrt_detg
    return Del


def inf_gen_eval(drift, cov, x, p=1):
    """ Apply the infinitesimal generator given SDE coefficients to a function p.

    Parameters:
        drift: the drift vector, a sympy expression
        cov: the covariance matrix, a sympy expression
        x: the variables of the coefficient functions and steady-state, sympy vector
        p: the smooth function which the operator is being applied to
    """

    L_drift = 0
    L_diff = 0
    N = x.shape[0]
    for i in range(N):
        L_drift += drift[i] * diff(p, x[i])
    for i in range(N):
        for j in range(N):
            L_diff += cov[i, j] * diff(p, x[i], x[j])
    inf_gen = L_drift + (1 / 2) * L_diff
    return inf_gen


def verify_steady_state(drift, cov, x, p=1):
    """ Verify the RHS side of the Fokker-Planck equation is zero for a given density, p, infinitesimal drift and
    covariance, at time=infinity. This means that the density p is a unnormalized steady-state of the process

    Parameters:
        drift: the drift vector, a sympy expression
        cov: the covariance matrix, a sympy expression
        x: the variables of the coefficient functions and steady-state, sympy vector
        p: the candidate steady-state density
    """
    fpe_drift = 0
    fpe_diff = 0
    N = x.shape[0]
    for i in range(N):
        fpe_drift += diff(drift[i] * p, x[i])
        for j in range(N):
            fpe_diff += diff(cov[i, j] * p, x[i], x[j])
    fpe = -fpe_drift + (1 / 2) * fpe_diff
    # fpe = simplify(fpe)
    return fpe


def hessian(f, x):
    d = x.shape[0]
    H = Matrix.zeros(d, d)
    for i in range(d):
        for j in range(d):
            H[i, j] = diff(f, x[i], x[j])
    return H


def grad_potential_np(U, param, aux=None):
    """ Lambdify the gradient of a potential function.

    Parameters:
        U : sympy expression, the potential function
        param : nx1 column matrix of the input to U
        aux : any auxilliary parameters defined in U
    """
    gradU = gradient(U, param)
    if aux is None:
        gradu_np = lambdify([param], gradU)
    else:
        gradu_np = lambdify([param, aux], gradU)
    return gradu_np


def set_potential_grad(gradu_np, p=None, d=2):
    """ Return a wrapper to the gradient U suitable to be passed to the Euler-Maruyama
    scheme.

    Parameters:
        gradu_np : function returned from grad_potential_np
        p : numpy array of aux parameters
        d : dimension of input to reshape by
    """
    if p is None:
        # def gradu_np1(t, x):
        return lambda x: gradu_np(x).reshape(d)
    else:
        # def gradu_np1(t, x):
        return lambda x: gradu_np(x, p).reshape(d)


def gradient(f, x):
    """ A simple implementation of computing a gradient of a function (sympy expression) with respect
    to its arguments (coordinates).

    Parameters:
        f : sympy expression, defining a function of the vector (nx1 column vector) x
        x : sympy matrix, a column vector of points in R^N

    Returns:
        column vector representing the gradient
    """
    N = x.shape[0]
    gf = Matrix.zeros(N, 1)
    for i in range(N):
        gf[i] = diff(f, x[i])
    return gf


def jacobian(f, x):
    """ Compute the Jacobian of an N-dimensional coordinate mapping of the
    d-dimensional parameters.

    Parameters:
        f: Nx1 matrix of the coordinates, each a function of 'x'
        x: dx1 matrix of the parameters

    Returns:
        Nxd matrix where each row is the gradient of a coordinate of the field f
    """
    N = f.shape[0]
    d = x.shape[0]
    J = Matrix.zeros(N, d)
    for i in range(N):
        J[i, :] = gradient(f[i], x).T
    return J


def divergence(field, x, detg=None):
    """ Compute the divergence of a vector field.

    Parameters:
        field : the vector field, a sympy nx1 column matrix
        x : the coordinates/arguments of the vector field
        detg : optional determinant of metric tensor g for manifold divergence
    """
    N = x.shape[0]
    div = 0
    if detg is None:
        for i in range(N):
            div += diff(field[i], x[i])
    else:
        for i in range(N):
            div += diff(sqrt(detg) * field[i], x[i])
        div = div / sqrt(detg)
    # Should we simplify in our functions ourselves?
    # div = simplify(div)
    return div


def divergence_matrix(M, x, detg=None):
    """ Compute the row divergence of a matrix.

    Parameters:
        M : the matrix to compute row divergence
        x : the coordinates/arguments of the vector field
        detg : optional determinant of metric tensor g for manifold divergence

    Returns:
        a sympy vector
    """
    d = x.shape[0]
    div = Matrix.zeros(d, 1)
    for i in range(d):
        div[i] = divergence(M[i, :], x, detg)
    # div = simplify(div)
    return div


def ito_formula(x, mu, sigma, f):
    """ Compute the drift and coefficient for a funciton of a multivariable process

    Parameters:
      x : the variables in the system
      mu : the drift vector
      sigma : the diffusion coefficient matrix (not the covariance matrix, it's square root!)
      f : the function that Ito's formula is applied to
    """
    N = x.shape[0]
    Covmat = sigma * sigma.T
    # Compute the gradient of f
    gradf = gradient(f, x)
    # gradf = zeros(N, 1)
    # for i in range(N):
    #     gradf[i, 0] = diff(f, x[i, 0])

    qv_term = 0
    for i in range(N):
        for j in range(N):
            qv_term += Covmat[i, j] * diff(f, x[i], x[j])
    Lf = gradf.T.dot(mu) + (1 / 2) * qv_term
    Lf = factor(Lf)
    diff_coef = simplify(gradf.T * sigma)
    return Lf, diff_coef


def compute_inf_gen(param, drift, cov, strf="f(t,x)", self_adjoint=False, spacetime=False):
    """ Compute the infinitesimal generator (over space) and display the Kolmogorov backward and
     forward PDEs for the process (assuming the infinitesimal generator is self-adjoint
     in the latter PDE).

     Parameters:
         param:
         drift: the infinitesimal drift coefficient vector
         cov: the infinitesimal covariance matrix (the square of the diffusion coefficient)
         strf: string of the test-function with its variables, e.g. f(t,x,y,z).
         self_adjoint: boolean for whether to print the FPE, assuming the infinitesimal
         generator is self-adjoint.
         spacetime: Boolean, if true a pseudo-Riemannian metric for spacetime is used and
         t is thus part of the parameters.
     """
    f = Function("f")
    t = symbols("t", real=True)
    inf_gen = 0.0
    d = param.shape[0]
    for i in range(d):
        inf_gen += drift[i] * f(*param).diff(param[i])
    for i in range(d):
        for j in range(d):
            inf_gen += (1 / 2) * cov[i, j] * f(*param).diff(param[i], param[j])
        # inf_gen = inf_gen.subs(f(*param), f)
    if not spacetime:
        inf_gen = inf_gen.subs(f(*param), f(t, *param))
    print("Kolmogorov Backward PDE")
    print(Eq(S(strf + ".diff(t)") + inf_gen, 0))
    if self_adjoint:
        print("Fokker-Planck equation (assuming self-adjointness)")
        print(Eq(S(strf + ".diff(t)"), inf_gen))
    return inf_gen
