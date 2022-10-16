# The module contains functions for learning the geometry of a manifold
# by observing Brownian motions assumed to run on it.
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from sympy import symbols, Matrix, solve, lambdify, sin, exp

from ManifoldBms.ManifoldBm.RiemannianGeo import IntrinsicBm, sample_path_coord
from ManifoldBms.ManifoldBm.sdes import estimate_cov
from ManifoldBms.ManifoldBm.sdes import sample_ensemble


def metric_tensor_decomp(proj, d=1, scale=1):
    """ Numerically estimate the metric tensor given a numerical orthogonal projection matrix at a point.

    Parameters:
        proj: numpy array (D,D), the numerical orthogonal projection matrix at a point x.
        d: int, the intrinsic dimension.

    Returns:
        (d,d) numpy array: the estimated metric tensor.
        float: volume density
        float: the sample mean of the first p=D-d eigenvalues.
    """
    extrinsic_dim = proj.shape[0]
    w, v = np.linalg.eigh(proj)
    p = extrinsic_dim - d
    jacobian = np.zeros((p, d))
    for i in range(p):
        entry = v[d+i, i]
        jacobian[i, :] = -v[:d, i].T
        # vector_zero = np.sum(v[d:, i]) - entry
        # vector_zero = np.abs(vector_zero) <= 0.01
        if entry != 0 :
            jacobian[i, :] = jacobian[i, :] / entry
    metric_tensor = np.eye(d) + jacobian.T.dot(jacobian)  # dot is safer shape wise
    volume_density = np.sqrt(np.linalg.det(metric_tensor))
    return metric_tensor, volume_density, np.mean(w[:p])


def learn_manifold_1d(ensembles, h=10 ** -5, d=1, scale_fun=None, lam=None):
    """ Given a collection of sample ensembles of a process, estimate the intrinsic metric tensor of the assumed
    underlying manifold.

    Returns tuple of metric tensor, volume density, and avg_eigenvalues, and x the points for g(x), vol(x), etc.
    """
    # Ensembles is either (m, N, n+1, D) or (m, n+1, N)
    # (num_ensembles, num_paths, timestep, dimension) for D-dimensional systems
    # (num_ensembles, timestep, num_paths) for 1-dimensional systems.
    ens_shape = ensembles.shape
    num_ensembles = ens_shape[0]
    D = ens_shape[3]
    metric_tensor = np.zeros((num_ensembles, d, d))
    volume_density = np.zeros(num_ensembles)
    avg_eigenvalues = np.zeros(num_ensembles)
    x = np.zeros((num_ensembles, d))
    if scale_fun is None:
        def scale_fun(w, lam=None):
            return 1
    for i in range(num_ensembles):
        x[i] = ensembles[i, 0][0, :d]
        scale = scale_fun(x[i], lam)
        proj = estimate_cov(ensembles[i], h)
        g, v, w = metric_tensor_decomp(proj, d, scale)
        metric_tensor[i] = g
        volume_density[i] = v
        avg_eigenvalues[i] = w
    return metric_tensor, volume_density, avg_eigenvalues, x


def learn_metric_tensor(y0, tn, bm, npaths, ntime, D, p, c=None, lam=None):
    """ Learn the metric tensor at a point y0 on synthetic model
    """
    d = bm.manifold.param.shape[0]
    drift, diffusion = bm.mu_np, bm.sigma_np
    # print("Size = "+str(d))
    # Input for single point estimator of any dimension ^ and below the function prototype
    Y = sample_ensemble(y0, tn, drift, diffusion, npaths, ntime, noise_dim=d, sys_dim=d)
    # For purposes of synthetic testing, we project upward
    # Project up to pretend to estimate from high dimension
    X = np.zeros((npaths, ntime + 1, D))
    for i in range(npaths):
        if len(Y.shape) == 2:  # d=1 then ensemble is (n+1, N)
            X[i] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[:, i], bm.aux, p)
        else:  # d>1 then ensemble is (N, n+1, d)
            X[i] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[i], bm.aux, p)
    Ph = estimate_cov(X, tn)
    if c is not None:
        c0 = c(y0, lam)
    else:
        c0 = 1
    g, volg, mwe = metric_tensor_decomp(Ph, d, c0)
    return g, volg, Y, X


def learn_metric_tensor_1d(y0, tn, bm, npaths, ntime, D, p, c=None, lam=None, plot=True):
    M = y0.shape[0]
    g = np.zeros(M)
    volg = np.zeros(M)
    d = bm.manifold.param.shape[0]
    tt = np.linspace(0, tn, ntime + 1)
    if plot:
        # fig = plt.figure(figsize=(6, 6))
        color = plt.cm.rainbow(np.linspace(0, 1, M))
    for i in range(M):
        g[i], volg[i], Y, X = learn_metric_tensor(y0[i], tn, bm, npaths, ntime, D, p, c, lam)
        if plot:
            plt.subplot(221)
            if d == 1:
                plt.plot(tt, Y, c=color[i], alpha=0.5)
            elif d == 2:
                for j in range(npaths):
                    plt.plot(Y[j][:, 0], Y[j][:, 1], c=color[i], alpha=0.5)
            plt.title("Intrinsic sample paths")
            for j in range(npaths):

                if D == 2:
                    plt.subplot(222)
                    plt.plot(X[j][:, 0], X[j][:, 1], c=color[i], alpha=0.5)
                elif D == 3:
                    ax = plt.subplot(222, projection='3d')
                    ax.plot3D(X[j][:, 0], X[j][:, 1], X[j][:, 2], c=color[i], alpha=0.5)
            plt.title("Extrinsic sample paths")
    return g, volg


def learn_metric_tensor_2d(y0, tn, bm, npaths, ntime, D, p, scale_fun=None, lam=None, plot=True):
    # print(y0[0].shape)
    M = y0[0].shape[0]
    d = bm.manifold.param.shape[0]
    gs = np.zeros((M, M, d, d))
    volg = np.zeros((M, M))
    if plot:
        # fig = plt.figure(figsize=(6, 6))
        color1 = plt.cm.rainbow(np.linspace(0, 1, M))
        color2 = plt.cm.rainbow(np.linspace(0, 1, M))
        color = np.zeros((M, M, 1, 4))
        for i in range(M):
            for j in range(M):
                # print(color1[i])
                color[i, j] = color1[i]
    for j in range(M):
        for k in range(M):
            u1 = y0[0][j, k]
            u2 = y0[1][j, k]
            y01 = np.array([u1, u2])
            g, gv, Y, X = learn_metric_tensor(y01, tn, bm, npaths, ntime, D, p, scale_fun, lam)
            gs[j, k] = g
            volg[j, k] = gv
            if plot:
                plt.subplot(221)
                if d == 2:
                    for l in range(npaths):
                        plt.plot(Y[l][:, 0], Y[l][:, 1], c=color[j, k], alpha=0.5)
                plt.title("Intrinsic sample paths")
                for l in range(npaths):

                    if D == 2:
                        plt.subplot(222)
                        plt.plot(X[l][:, 0], X[l][:, 1], c=color[j, k], alpha=0.5)
                    elif D > 2:
                        ax = plt.subplot(222, projection='3d')
                        ax.plot3D(X[l][:, 0], X[l][:, 1], X[l][:, 2], c=color[j, k], alpha=0.5)
                plt.title("Extrinsic sample paths")

    return gs, volg


def synthetic_test(x0, tn, drift, diffusion, bm, scale_fun, lam, nens, npaths, ntime, D, d, plot=True):
    ensembles = np.zeros((nens, npaths, ntime + 1, D))
    color = None
    ax = None
    if plot:
        fig, ax = plt.subplots(2, 2)
        color = plt.cm.rainbow(np.linspace(0, 1, nens))
    for i in range(nens):
        # Efficiently simulate in low dimension
        Y = sample_ensemble(x0[i], tn, drift, diffusion, npaths, ntime, d, d)
        if plot:
            ax[0, 0].plot(np.linspace(0, tn, ntime + 1), Y, c=color[i], alpha=0.5)
        # Convert to high dimension
        X = np.zeros((npaths, ntime + 1, D))
        for j in range(npaths):
            if len(Y.shape) == 2:  # d=1 then ensemble is (n+1, N)
                X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[:, j])
            else:  # d>1 then ensemble is (N, n+1, d)
                X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[j])
            if plot:
                ax[0, 1].plot(X[j][:, 0], X[j][:, 1], c=color[i], alpha=0.5)

        ensembles[i, :] = X
    if plot:
        ax[0, 0].set_title("Intrinsic sample paths")
        ax[0, 1].set_title("Extrinsic sample paths")
    metric_tensor, volume_density, avg_eigenvalues, xx = learn_manifold_1d(ensembles, tn, d, scale_fun, lam)

    exact_vol = lambdify(param, bm.manifold.vol)
    # curve = lambdify(param, F)
    exact = exact_vol(xx)
    mse_arc_length_density = np.mean((exact - volume_density) ** 2)
    estimated_arc_length = cumulative_trapezoid(volume_density, x=xx[:, 0], initial=0)
    true_arc_length = cumulative_trapezoid(exact[:, 0], xx[:, 0], initial=0)
    mse = np.mean((true_arc_length - estimated_arc_length) ** 2)
    print("Mean Square Error of arc-length density estimation = " + str(mse_arc_length_density))
    print("Mean Square Error of arc-length estimation = " + str(mse))
    print("Total arc length = " + str(true_arc_length[-1]))

    if plot:
        # Plot the volume-density
        # fig, ax = plt.subplots(2)
        ax[1, 0].plot(xx, volume_density)
        ax[1, 0].plot(xx, exact)
        ax[1, 0].set_title("Arc-length density")
        ax[1, 0].legend(["Estimated", "Exact"])
        ax[1, 1].plot(xx, estimated_arc_length)
        ax[1, 1].plot(xx, true_arc_length)
        ax[1, 1].set_title("Arc-length")
        ax[1, 1].legend(["Estimated", "Exact"])
        plt.show()
    return mse_arc_length_density


def synthetic_test_1d(param, x0, tn, drift, diffusion, bm, scale_fun, lam, nens, npaths, ntime, D, d, plot=True):
    # Heavily depends on shape of x0
    # if x0 is 1-d array, no changes need to be made.
    # if x0 is 2d, we really need a mesh for best visualization.
    # so the user must pash the mesh.
    ensembles = np.zeros((nens, npaths, ntime + 1, D))
    tt = np.linspace(0, tn, ntime + 1)
    ax = None
    color = None
    if plot:
        # fig, ax = plt.subplots(2, 2)
        fig = plt.figure(figsize=(6, 6))

        # set up a figure twice as wide as it is tall
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # # set up the axes for the first plot

        color = plt.cm.rainbow(np.linspace(0, 1, nens))
    for i in range(nens):
        # Efficiently simulate in low dimension: can only be 1 or 2 for 3d.
        Y = sample_ensemble(x0[i], tn, drift, diffusion, npaths, ntime, d, d)
        if plot:
            plt.subplot(221)
            if d == 1:
                plt.plot(tt, Y, c=color[i], alpha=0.5)
            elif d == 2:
                for j in range(npaths):
                    plt.plot(Y[i][:, 0], Y[i][:, 1], c=color[i], alpha=0.5)
            plt.title("Intrinsic sample paths")
        # Convert to high dimension
        X = np.zeros((npaths, ntime + 1, D))
        for j in range(npaths):

            if len(Y.shape) == 2:  # d=1 then ensemble is (n+1, N)

                X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[:, j])
            else:  # d>1 then ensemble is (N, n+1, d)
                X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[j])
            if plot:
                if D == 2:
                    plt.subplot(222)
                    plt.plot(X[j][:, 0], X[j][:, 1], c=color[i], alpha=0.5)
                elif D == 3:
                    # plt.subplot(222)
                    # plt.plot(X[j][:, 1], X[j][:, 2], c=color[i], alpha=0.5)

                    ax = plt.subplot(222, projection='3d')
                    ax.plot3D(X[j][:, 0], X[j][:, 1], X[j][:, 2], c=color[i], alpha=0.5)
                plt.title("Extrinsic sample paths")
        ensembles[i, :] = X
    metric_tensor, volume_density, avg_eigenvalues, xx = learn_manifold_1d(ensembles, tn, d, scale_fun, lam)
    exact_vol = lambdify(param, bm.manifold.vol)
    # curve = lambdify(param, F)
    exact = exact_vol(xx)
    if len(exact.shape) == 0:
        exact = np.ones(xx.shape[0]) * exact
        exact = exact.reshape((exact.shape[0], 1))
    mse_arc_length_density = np.mean((exact - volume_density) ** 2)
    estimated_arc_length = cumulative_trapezoid(volume_density, x=xx[:, 0], initial=0)
    true_arc_length = cumulative_trapezoid(exact[:, 0], xx[:, 0], initial=0)
    mse = np.mean((true_arc_length - estimated_arc_length) ** 2)
    print("Mean Square Error of arc-length density estimation = " + str(mse_arc_length_density))
    print("Mean Square Error of arc-length estimation = " + str(mse))
    print("Total arc length = " + str(true_arc_length[-1]))

    if plot:
        # Plot the volume-density
        # fig, ax = plt.subplots(2)
        plt.subplot(223)
        plt.plot(xx, volume_density)
        plt.plot(xx, exact)
        plt.title("Arc-length density")
        plt.legend(["Estimated", "Exact"])
        plt.subplot(224)
        plt.plot(xx, estimated_arc_length)
        plt.plot(xx, true_arc_length)
        plt.title("Arc-length")
        plt.legend(["Estimated", "Exact"])
        plt.show()
    return mse_arc_length_density


def synthetic_test_2d(param, grid, tn, drift, diffusion, bm, scale_fun, lam, nens, npaths, ntime, D, d, plot=True):
    # Heavily depends on shape of x0
    # if x0 is 1-d array, no changes need to be made.
    # if x0 is 2d, we really need a mesh for best visualization.
    # so the user must pash the mesh.
    M = grid[0].shape[0]
    nens = M * M
    ensembles = np.zeros((nens, npaths, ntime + 1, D))
    ax = None
    color = None
    if plot:
        # fig, ax = plt.subplots(2, 2)
        fig = plt.figure(figsize=(6, 6))

        # set up a figure twice as wide as it is tall
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # # set up the axes for the first plot

        color = plt.cm.rainbow(np.linspace(0, 1, nens))
    for i in range(nens):
        # Efficiently simulate in low dimension: can only be 1 or 2 for 3d.
        if d == 1:
            Y = sample_ensemble(x0[i], tn, drift, diffusion, npaths, ntime, d, d)
        elif d == 2:
            Y = sample_ensemble(x0[i], tn, drift, diffusion, npaths, ntime, d, d)
        if plot:
            plt.subplot(221)
            if d == 1:

                plt.plot(np.linspace(0, tn, ntime + 1), Y, c=color[i], alpha=0.5)
            elif d == 2:
                for j in range(npaths):
                    plt.plot(Y[i][:, 0], Y[i][:, 1], c=color[i], alpha=0.5)
            plt.title("Intrinsic sample paths")
        # Convert to high dimension
        X = np.zeros((npaths, ntime + 1, D))
        for j in range(npaths):

            if len(Y.shape) == 2:  # d=1 then ensemble is (n+1, N)

                X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[:, j])
            else:  # d>1 then ensemble is (N, n+1, d)
                X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[j])
            if plot:
                if D == 2:
                    plt.subplot(222)
                    plt.plot(X[j][:, 0], X[j][:, 1], c=color[i], alpha=0.5)
                elif D == 3:
                    # plt.subplot(222)
                    # plt.plot(X[j][:, 1], X[j][:, 2], c=color[i], alpha=0.5)

                    ax = plt.subplot(222, projection='3d')
                    ax.plot3D(X[j][:, 0], X[j][:, 1], X[j][:, 2], c=color[i], alpha=0.5)
                plt.title("Extrinsic sample paths")
        ensembles[i, :] = X
    metric_tensor, volume_density, avg_eigenvalues, xx = learn_manifold_1d(ensembles, tn, d, scale_fun, lam)
    exact_vol = lambdify(param, bm.manifold.vol)
    # curve = lambdify(param, F)
    exact = exact_vol(xx)
    if len(exact.shape) == 0:
        exact = np.ones(xx.shape[0]) * exact
        exact = exact.reshape((exact.shape[0], 1))
    mse_arc_length_density = np.mean((exact - volume_density) ** 2)
    estimated_arc_length = cumulative_trapezoid(volume_density, x=xx[:, 0], initial=0)
    true_arc_length = cumulative_trapezoid(exact[:, 0], xx[:, 0], initial=0)
    mse = np.mean((true_arc_length - estimated_arc_length) ** 2)
    print("Mean Square Error of arc-length density estimation = " + str(mse_arc_length_density))
    print("Mean Square Error of arc-length estimation = " + str(mse))
    print("Total arc length = " + str(true_arc_length[-1]))

    if plot:
        # Plot the volume-density
        # fig, ax = plt.subplots(2)
        plt.subplot(223)
        plt.plot(xx, volume_density)
        plt.plot(xx, exact)
        plt.title("Arc-length density")
        plt.legend(["Estimated", "Exact"])
        plt.subplot(224)
        plt.plot(xx, estimated_arc_length)
        plt.plot(xx, true_arc_length)
        plt.title("Arc-length")
        plt.legend(["Estimated", "Exact"])
    return mse_arc_length_density


if __name__ == "__main__":
    tn = 10 ** -9
    a = -3
    b = 3
    nens = 50
    npaths = 50
    ntime = 20
    x0 = np.linspace(a, b, nens)
    lam = np.array([0.2, 0.5])


    def scale_fun(w, l):
        return np.sqrt(l[0] * np.abs(w) + l[1])


    # Setting up the synthetic process
    x, y = symbols("x y", real=True)
    F = exp(-sin(x) ** 2)
    f = F - y
    # Now we can set up an instrinsic BM
    param = Matrix([x])
    chart = Matrix([x, solve(f, y)[-1]])
    d = param.shape[0]
    D = chart.shape[0]
    print("Chart")
    print(chart)
    bm = IntrinsicBm(param, chart)
    drift, diffusion = bm.get_bm_coefs(d)
    synthetic_test(x0, tn, drift, diffusion, bm, None, None, nens, npaths, ntime, D, d, True)


    def cost_function(lam):
        mse = synthetic_test(x0, tn, drift, diffusion, bm, scale_fun, lam, nens, npaths, ntime, D, d, False)
        # print("Cost function = "+str(mse))
        return mse


    opt_lam = minimize(cost_function, lam, method="L-BFGS-B", bounds=[(0, 4), (0, 4)])
    print(opt_lam)
    opt_lam = opt_lam["x"]
    synthetic_test(x0, tn, drift, diffusion, bm, scale_fun, opt_lam, nens, npaths, ntime, D, d, True)

    # ensembles = np.zeros((nens, npaths, ntime+1, D))
    # fig, ax = plt.subplots(2, 2)
    # color = plt.cm.rainbow(np.linspace(0, 1, nens))
    # for i in range(nens):
    #     # Efficiently simulate in low dimension
    #     Y = sample_ensemble(x0[i], tn, drift, diffusion, npaths, ntime, d, d)
    #     ax[0,0].plot(np.linspace(0, tn, ntime+1),Y, c=color[i], alpha=0.5)
    #     # Convert to high dimension
    #     X = np.zeros((npaths, ntime + 1, D))
    #     for j in range(npaths):
    #         if len(Y.shape) == 2:  # d=1 then ensemble is (n+1, N)
    #             X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[:, j])
    #         else:  # d>1 then ensemble is (N, n+1, d)
    #             X[j] = sample_path_coord(bm.manifold.param, bm.manifold.chart, Y[j])
    #         ax[0,1].plot(X[j][:, 0], X[j][:, 1], c=color[i], alpha=0.5)
    #
    #     ensembles[i, :] = X
    # ax[0,0].set_title("Intrinsic sample paths")
    # ax[0, 1].set_title("Extrinsic sample paths")
    # metric_tensor, volume_density, avg_eigenvalues, xx = learn_manifold(ensembles, tn, d, scale_fun, lam)
    #
    # exact_vol = lambdify(param, bm.manifold.vol)
    # curve = lambdify(param, F)
    # exact = exact_vol(xx)
    # mse_arc_length_density = np.mean((exact-volume_density)**2)
    # estimated_arc_length = cumulative_trapezoid(volume_density, x = xx[:,0], initial=0)
    # true_arc_length = cumulative_trapezoid(exact[:,0], xx[:,0], initial=0)
    # mse = np.mean((true_arc_length-estimated_arc_length)**2)
    # print("Mean Square Error of arc-length density estimation = " + str(mse_arc_length_density))
    # print("Mean Square Error of arc-length estimation = "+str(mse))
    # print("Total arc length = "+str(true_arc_length[-1]))
    #
    # # Plot the volume-density
    # # fig, ax = plt.subplots(2)
    # ax[1, 0].plot(xx, volume_density)
    # ax[1, 0].plot(xx, exact)
    # ax[1, 0].set_title("Arc-length density")
    # ax[1, 0].legend(["Estimated", "Exact"])
    # ax[1, 1].plot(xx, estimated_arc_length)
    # ax[1, 1].plot(xx, true_arc_length)
    # ax[1, 1].set_title("Arc-length")
    # ax[1, 1].legend(["Estimated", "Exact"])
    # plt.show()


def synthetic_test1(param, x0, tn, bm, npaths, ntime, D, p, scale_fun, lam, plot=True):
    # if plot:
        # fig, ax = plt.subplots(2, 2)
        # fig = plt.figure(figsize=(6, 6))
    metric_tensor, volume_density = learn_metric_tensor_1d(x0, tn, bm, npaths, ntime, D, p, scale_fun, lam)
    exact_vol = lambdify(param, bm.manifold.vol)
    # curve = lambdify(param, F)
    exact = exact_vol(x0)
    if len(exact.shape) == 0:
        exact = np.ones(x0.shape[0]) * exact
        exact = exact.reshape((exact.shape[0], 1))
    mse_arc_length_density = np.mean((exact - volume_density) ** 2)
    estimated_arc_length = cumulative_trapezoid(volume_density, x=x0, initial=0)
    true_arc_length = cumulative_trapezoid(exact, x0, initial=0)
    mse = np.mean((true_arc_length - estimated_arc_length) ** 2)
    print("Mean Square Error of arc-length density estimation = " + str(mse_arc_length_density))
    print("Mean Square Error of arc-length estimation = " + str(mse))
    print("Total arc length = " + str(true_arc_length[-1]))

    if plot:
        # Plot the volume-density
        # fig, ax = plt.subplots(2)
        plt.subplot(223)
        plt.plot(x0, volume_density, alpha=0.5)
        plt.plot(x0, exact, alpha=0.5)
        plt.title("Arc-length density")
        plt.legend(["Estimated", "Exact"])
        plt.subplot(224)
        plt.plot(x0, estimated_arc_length, alpha=0.5)
        plt.plot(x0, true_arc_length, alpha=0.5)
        plt.title("Arc-length")
        plt.legend(["Estimated", "Exact"])
        # plt.show()
    return mse_arc_length_density


def synthetic_test2(param, grid, tn, bm, npaths, ntime, D, p, scale_fun, lam, plot=True):
    # if plot:
    #
    #     fig = plt.figure(figsize=(6, 6))
    metric_tensor, volume_density = learn_metric_tensor_2d(grid, tn, bm, npaths, ntime, D, p, scale_fun, lam, plot)

    aux = None
    # Now for the exact values
    if aux is None:
        vol_np1 = lambdify([param], bm.manifold.vol)
    else:
        vol_np1 = lambdify([param, aux], bm.manifold.vol)

    def exact_vol(x1, x2):
        print(x1.shape)
        M = x1.shape[0]
        V = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                u1 = x1[i, j]
                u2 = x2[i, j]
                xx = np.array([u1, u2])
                if p is None:
                    v = vol_np1(xx)
                else:
                    v = vol_np1(xx, p)
                V[i, j] = v
        return V

    # exact_vol = lambdify(param, bm.manifold.vol)
    # curve = lambdify(param, F)
    exact = exact_vol(grid[0], grid[1])

    if len(exact.shape) == 0:
        exact = np.ones(grid.shape[0]) * exact
        exact = exact.reshape((exact.shape[0], 1))
    mse_arc_length_density = np.mean((exact - volume_density) ** 2)

    # Need to estimate surface-area by computing cumulative double integrals...
    M = grid[0].shape[0]
    estimated_arc_length = np.zeros((M, M))
    true_arc_length = np.zeros((M, M))
    x0 = grid[0][:,0]
    y0 = grid[1][0, :]
    for i in range(M):
        estimated_arc_length[i, :] = cumulative_trapezoid(volume_density[i,:], x=y0, initial=0)
        true_arc_length[i, :] = cumulative_trapezoid(exact[i,:], x=y0, initial=0)
    for i in range(M):
        estimated_arc_length[:, i] = cumulative_trapezoid(estimated_arc_length[:,i], x=x0, initial=0)
        true_arc_length[:, i] = cumulative_trapezoid(true_arc_length[:,i], x=x0, initial=0)

    mse = np.mean((true_arc_length - estimated_arc_length) ** 2)
    print("Mean Square Error of surface area density estimation = " + str(mse_arc_length_density))
    print("Mean Square Error of surface area estimation = " + str(mse))
    print("Total estimated surface area = " + str(estimated_arc_length[-1, -1]))
    print("Total true surface area = " + str(true_arc_length[-1,-1]))

    if plot:
        # Plot the volume-density function
        ax = plt.subplot(223, projection='3d')
        ax.plot_surface(grid[0], grid[1], volume_density)
        ax.plot_surface(grid[0], grid[1], exact, alpha=0.5)
        ax.set_title("Surface-Area density")
        # plt.legend(["Estimated", "Exact"])

        ax = plt.subplot(224, projection="3d")
        ax.plot_surface(grid[0], grid[1], estimated_arc_length)
        ax.plot_surface(grid[0], grid[1], true_arc_length, alpha=0.5)
        ax.set_title("Surface area")
        # plt.legend(["Estimated", "Exact"])
        # plt.show()
    return mse_arc_length_density
