import matplotlib.pyplot as plt
# Numerical estimation of a Riemannian metric
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import minimize
from sympy import lambdify, Piecewise
from sympy import simplify, sqrt, symbols, Matrix, sin, cos, integrate, pi

from ManifoldBms.ManifoldBm.sdes import euler_maruyama, sample_ensemble, estimate_cov
from SympyRoutines import jacobian, divergence_matrix, gradient


def learn_synthetic_metric(y0, bm, extrinsic_param, f, tn=10 ** -3, npaths=1000, ntime=10):
    # Efficient simulating in low dimension
    noise_dim = bm.manifold.param.shape[0]
    d = noise_dim
    sys_dim = noise_dim
    D = extrinsic_param.shape[0]
    # Numerical coefficients
    drift, diffusion = bm.get_bm_coefs(d)
    # Extrinsic projection matrix
    n = gradient(f, extrinsic_param)
    n = n / n.norm()
    I = Matrix.eye(D)
    P = I - n * n.T
    P_at_x = lambdify([extrinsic_param], P)

    # The manifold and the SDEs
    print("==========~Manifold and SDE coefficients~===========")
    print(bm.manifold)
    print(bm)
    print("Orthogonal projection")
    print(P)

    if type(y0) is tuple:
        print(y0[0].shape)
        M = y0[0].shape[0]
        gs = np.zeros((M, M, d, d))
        volg = np.zeros((M, M))
        for j in range(M):
            for k in range(M):
                u1 = y0[0][j, k]
                u2 = y0[1][j, k]
                y01 = np.array([u1, u2])
                Y = sample_ensemble(y01, tn, drift, diffusion, npaths, ntime, noise_dim, sys_dim)
                # Project up to pretend to estimate from high dimension
                X = np.zeros((npaths, ntime + 1, D))
                for i in range(npaths):
                    if len(Y.shape) == 2:  # d=1 then ensemble is (n+1, N)
                        X[i] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[:, i])
                    else:  # d>1 then ensemble is (N, n+1, d)
                        X[i] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[i])
                x0 = X[0][0, :]

                # We are going to uninterestingly learn the low dimensional geometry, a line
                Ph = estimate_cov(X, tn)
                g = estimate_metric_tensor(Ph, d)

                # Now we have an ensemble of sample paths on a manifold
                # print("Shape of ensemble")
                # print(X.shape)
                # print("Estimate of orthogonal projection at x = " + str(x0))
                # print(Ph)
                # print("True orthogonal projection at x = " + str(x0))
                # print(P_at_x(x0))
                # print("Estimated metric tensor")
                # print(g)
                gs[j, k] = g
                volg[j, k] = np.sqrt(np.abs(np.linalg.det(g)))
                # fig1 = plt.figure()
                # plt.plot(Y)
                # plt.show()
    else:  # when d=1 dimension
        M = y0.shape[0]
        gs = np.zeros((M, d, d))
        volg = np.zeros(M)
        for j in range(M):
            if len(y0.shape) > 1:
                Y = sample_ensemble(y0[j, :], tn, drift, diffusion, npaths, ntime, noise_dim, sys_dim)
            else:
                Y = sample_ensemble(y0[j], tn, drift, diffusion, npaths, ntime, noise_dim, sys_dim)

            # Project up to pretend to estimate from high dimension
            X = np.zeros((npaths, ntime + 1, D))
            for i in range(npaths):
                if len(Y.shape) == 2:  # d=1 then ensemble is (n+1, N)
                    X[i] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[:, i])
                else:  # d>1 then ensemble is (N, n+1, d)
                    X[i] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[i])
            x0 = X[0][0, :]
            # We are going to uninterestingly learn the low dimensional geometry, a line
            Ph = estimate_cov(X, tn)
            g = estimate_metric_tensor(Ph, d)

            # Now we have an ensemble of sample paths on a manifold
            # print("Shape of ensemble")
            # print(X.shape)
            # print("Estimate of orthogonal projection at x = " + str(x0))
            # print(Ph)
            # print("True orthogonal projection at x = " + str(x0))
            # print(P_at_x(x0))
            # print("Estimated metric tensor")
            # print(g)
            gs[j] = g
            volg[j] = np.sqrt(np.abs(np.linalg.det(g)))
            # Causes error if y is 1 dim array of carrying two values representing y0 in 2d
            if len(y0.shape) == 1:
                gs = gs.reshape(M)
        # fig1 = plt.figure()
        # plt.plot(Y)
        # plt.show()
    # Plot the high dimensional observations
    if D == 2:

        fig = plt.figure()
        for i in range(npaths):
            plt.plot(X[i][:, 0], X[i][:, 1])
        plt.show()
    else:
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        for i in range(npaths):
            ax.plot3D(X[i][:, 0], X[i][:, 1], X[i][:, 2])
        plt.show()
    return gs, volg


# Now we can back out the metric tensor as follows.
def estimate_metric_tensor(orthog_proj, d=2, c=None, pr=False):
    D = orthog_proj.shape[0]
    p = D - d
    w, v = np.linalg.eigh(orthog_proj)
    if pr:
        print("p-first eigenvalues")
        print(w[:p])
    J = -v[:d, :p].T
    if c is not None:
        g = np.eye(d) + c * J.T * J
    else:
        g = np.eye(d) + J.T * J
    return g


def estimate_metric1(ens, h, d=2):
    P = estimate_cov(ens, h)
    D = P.shape[0]
    w, v = np.linalg.eigh(P)
    p = D - d
    J = -v[:d, :p].T
    g = np.eye(d) + J.T * J
    return g


def learn_metric(param, chart, f, x0, tn, p, aux, ntime, npaths, noise_dim, sys_dim):
    bm = IntrinsicBm(param, chart, aux=aux)
    print("Manifold:")
    print(bm.manifold)
    print("Ito Drift and Diffusion Coefficients of BM SDE on manifold:")
    print(bm)
    drift, diffusion = bm.get_bm_coefs(d=2, p=p)
    # Now simulate ensemble
    ens = sample_ensemble(x0, tn, drift, diffusion, npaths, ntime, noise_dim, sys_dim)
    # We need extrinsic sample path!
    # We need to transform to the extrinsic setting
    extrinsic_ens = np.zeros((npaths, ntime + 1, 3))
    for i in range(npaths):
        extrinsic_ens[i] = sample_path_coord(param, chart, ens[i], aux, p)
    # Finally we estimate the orthogonal projection matrix
    cov = estimate_cov(extrinsic_ens, tn)
    # Compare the results
    x, y, z = symbols("x y z", real=True)
    # f = (x / a) ** 2 + (y / b) ** 2 - z

    n = gradient(f, Matrix([x, y, z]))
    n = n / n.norm()
    P = Matrix.eye(3) - n * n.T
    # Numpyfying the Orthogonal Projection and Metric tensor for verification
    P_np = lambdify([Matrix([x, y, z]), aux], P)
    g_np = lambdify([param, aux], bm.manifold.g)
    ex_x0 = extrinsic_ens[0][0, :]
    print(ex_x0.shape)
    print("Estimated Orthogonal Projection")
    print(cov)
    print("True Orthogonal Projection")
    print(P_np(ex_x0, p))
    print(np.linalg.norm(P_np(ex_x0, p) - cov))

    g_est = estimate_metric_tensor(cov, d=2)
    print("estimated metric tensor")
    print(g_est)
    print("true metric tensor")
    print(g_np(x0, p))
    #
    # ===============================
    # 3d plot of the uniform sample
    # ===============================
    # set up the axes for the first plot
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    # Plot the surface
    for i in range(npaths):
        ax.plot3D(extrinsic_ens[i][:, 0], extrinsic_ens[i][:, 1], extrinsic_ens[i][:, 2], "blue", alpha=0.5)
    # ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.4)
    ax.set_title("BM ensemble on manifold")
    plt.show()


def get_piecewise_drift(mu, index=0):
    """ Some drifts turn out piecewise due to the nature of sympy. Use this function to extract the relevant
    expression.
    """
    u = Matrix(simplify(mu))
    if type(u[index]) is Piecewise:
        u[index] = u[index].args[1][0]
    return u


# This function is for mapping the parameters x to xi(x)
def sample_path_coord(param, coord, xt, aux=None, p=None):
    """ Pass the parameter sample path to the coordinate transformation.

    (Parameters):
    param: sympy object defining parameters
    coord: sympy object defining the coordinate transformation
    xt: sample path returned from euler_maruyama_2d
    aux: sympy Matrix for auxiliary parameters in the metric tensor
    p: numpy array for the numerical values of any auxiliary parameters in the equations
    """
    D = coord.shape[0]
    n = xt.shape[0]
    d = param.shape[0]
    if aux is None:

            coord_np = lambdify([param], coord)
    else:
        if D == 2:
            coord_np = lambdify([param, aux], coord)
        else:
            coord_np = lambdify([param, aux], coord)
    if p is None:
        # if D == 2:
        #     output = coord_np(xt.T).T.reshape(n, D)
        if D >= 2:
            if len(xt.shape) == 1:
                output = coord_np(xt.reshape((d, n))).reshape(D, n).T
            else:
                output = coord_np(xt.T).T.reshape(n, D)
        else:
            output = coord_np(xt.T, p).T.reshape(n, D)
    return output


class mle(object):
    def __init__(self, h, a, b, c, d, bds):
        self.h = h
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.bds = bds

    def z(self, theta):
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        return dblquad(lambda u, v: self.h(np.array([u, v]), theta), a, b, c, d)[0]

    def ll(self, theta, X):
        n = X.shape[0]
        return -(np.sum(np.log(self.h(X, theta)), axis=0) - n * np.log(self.z(theta)))

    def fit(self, X, theta0):
        w = minimize(self.ll, theta0, args=X)
        print("Unconstrained optimization")
        print(w)
        print("\n")
        return minimize(self.ll, theta0, args=X, method="L-BFGS-B", bounds=self.bds)


class Manifold(object):
    # How should we implement Manifold in python?
    # The user supplies the intrinsic coordinates and the local chart
    # The metric tensor is automatically computed but is optionally simplified.
    #
    # to do: adding more methods for computing volume measure, geodesics, arc length
    def __init__(self, param, chart, simp=True):
        """ Define a manifold intrinsically using a (local) parameterization.

        Parameters:
            param: a sympy dx1 matrix, the intrinsic coordinates of the manifold
            chart: a sympy Dx1 matrix, the extrinsic coordinates of the manifold, D>d
            simp: boolean for simplifying the metric tensor algebraically using sympy

        The attributes are g, and vol for the metric tensor and volume measure.
        """
        self.g = None
        self.vol = None
        self._det_g = None
        self.param = param
        self.chart = chart
        self.simp = simp
        self._compute_metric_tensor()
        self._volume_measure()

    def __str__(self):
        s = "Metric tensor:\n" + str(self.g) + "\nVolume density = " + str(self.vol)
        return s

    def simplify_terms(self, term):
        if self.simp:
            term = simplify(term)
        return term

    def _compute_metric_tensor(self):
        """ Compute the metric tensor of a surface defined via a parameterization through the equation
        g = J^T J, where J is the (non-square) Jacobian matrix of the parameterization.
        """
        j = jacobian(self.chart, self.param)
        g = j.T * j
        g = self.simplify_terms(g)
        self.g = g
        return None

    def _volume_measure(self):
        """ Compute the volume measure intrinsically. This is equal to sqrt(det g), where g is the metric
        tensor.
        """
        # The determinant will be used in other classes, like SDE, so we should store it.
        self._det_g = self.g.det()
        self._det_g = self.simplify_terms(self._det_g)
        vol = sqrt(self._det_g)
        vol = self.simplify_terms(vol)
        self.vol = vol
        return None


class IntrinsicBm(object):
    def __init__(self, param=None, chart=None, aux=None, manifold=None, simp=True):
        self.sigma_np = None
        self.mu_np = None
        self.sigma = None
        self.mu = None
        self.aux = aux
        if manifold is None:
            self.manifold = Manifold(param, chart, simp)
        else:
            self.manifold = manifold
        self.drift_sym = None
        self.diff_sym = None
        self.cov_sym = None
        self.detg = None
        self._compute_detinvsqr()
        self._compute_drift()
        self._numpify(aux)
        w1, w = self.get_bm_coefs(d=self.manifold.param.shape[0])

    def __str__(self):
        s = "Drift coefficient:\n" + str(self.drift_sym) + "\nDiffusion coefficientt:\n" + str(self.diff_sym)
        return s

    def _compute_detinvsqr(self):
        g = self.manifold.g
        """ Compute the determinant of a metric tensor, its inverse and the
        Cholesky decomposition of the inverse metric tensor.
        """
        if g.is_diagonal():
            print("Computing determinant, inverse of metric tensor and its sqrt...")
            d = g.shape[0]
            detg = 1
            ginv = Matrix.zeros(d, d)
            sqrt_ginv = Matrix.zeros(d, d)
            for i in range(d):
                detg *= g[i, i]
                ginv[i, i] = 1 / g[i, i]
                sqrt_ginv[i, i] = sqrt(ginv[i, i])
        else:
            print("Computing determinant of metric tensor...")
            detg = g.det()
            detg = simplify(detg)
            print("Computing inverse of metric tensor...")
            ginv = g.inv()
            ginv = simplify(ginv)
            print("Computing square root matrix of coefficient...")
            sqrt_ginv = ginv * detg
            sqrt_ginv = sqrt_ginv.cholesky(hermitian=False)
            sqrt_ginv = simplify(sqrt_ginv)
            sqrt_ginv = sqrt_ginv / sqrt(detg)
        self.cov_sym = ginv
        self.diff_sym = sqrt_ginv
        self.detg = detg
        return None

    def _compute_drift(self):
        """ The drift of a BM on manifold is 1/2 the manifold-divergence of the inverse
        metric tensor.

        Returns:
            sympy matrix, the vector containing each drift coefficient for the system.
        """
        print("Computing drifts...")
        mu = divergence_matrix(self.cov_sym, self.manifold.param, self.detg)
        mu = mu / 2
        mu = self.manifold.simplify_terms(mu)
        self.drift_sym = get_piecewise_drift(mu)

        return None

    def _numpify(self, aux):
        if aux is None:
            # Square root (Cholesky decomp) of g^-1
            sqrt_ginv_np = lambdify([self.manifold.param], self.diff_sym)

            # This will need to be reshaped for our Euler-Maruyama Scheme
            mu_np1 = lambdify([self.manifold.param], self.drift_sym)
        else:
            # Square root (Cholesky decomp) of g^-1
            sqrt_ginv_np = lambdify([self.manifold.param, aux], self.diff_sym)

            # This will need to be reshaped for our Euler-Maruyama Scheme
            mu_np1 = lambdify([self.manifold.param, aux], self.drift_sym)
        self.mu = mu_np1
        self.sigma = sqrt_ginv_np
        # return mu_np1, sqrt_ginv_np

    # Consider rewriting this as a optional args* list?
    def get_bm_coefs(self, d=2, p=None):
        """ Pass the lambdified functions and return wrappers properly
        defined to be passed to 'euler_maruyama_2d'

        (Parameters)
        d : the dimensions of the system, i.e. numbers of variables in parameterization
        p : array, the auxilliary parameters, defaults to None
        """

        if p is None:

            def mu_np(x):
                return self.mu(x).reshape(d)

            def sigma_np(x):
                return self.sigma(x)
        else:
            def mu_np(x):
                return self.mu(x, p).reshape(d)

            def sigma_np(x):
                return self.sigma(x, p)
        self.mu_np = mu_np
        self.sigma_np = sigma_np
        return mu_np, sigma_np

    def uniform_sample(self, N, x0, tn, k=1000, theta=None):
        n = N + k - 2
        d = x0.shape[0]
        mu, sigma = self.get_bm_coefs(d, p=theta)
        xt = euler_maruyama(x0, tn, mu, sigma, n, d, d)
        return xt[(n - N):]

    def numpify_volume(self):
        vol = lambdify([self.manifold.param, self.aux], self.manifold.vol)

        def vol_np(x, theta):
            if len(x.shape) == 1:
                return vol(x, theta)
            else:
                n = x.shape[0]
                v = np.zeros(n)
                for i in range(n):
                    v[i] = vol(x[i, :], theta)
                return v

        return vol_np


if __name__ == "__main__":
    # Testing the paraboloid
    u, v = symbols("u v", real=True)
    x = Matrix([u, v])
    z = u ** 2 + v ** 2
    X = Matrix([u, v, z])
    m = Manifold(x, X, True)
    print("Paraboloid:")
    print(m)

    mbm = IntrinsicBm(manifold=m)
    print(mbm)

    # Testing the sphere
    theta, phi = symbols("theta phi", real=True, positive=True)
    x = Matrix([theta, phi])
    x1 = cos(phi) * sin(theta)
    x2 = sin(phi) * sin(theta)
    x3 = cos(theta)
    X = Matrix([x1, x2, x3])
    m = Manifold(x, X, True)
    print("\nSphere:")
    print(m)
    print("Total surface area of sphere:")
    v1 = integrate(m.vol, (theta, 0, pi), (phi, 0, 2 * pi))
    print(v1)

    mbm = IntrinsicBm(manifold=m)
    print(mbm)
