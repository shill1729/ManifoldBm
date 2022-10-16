# ManifoldBm
This package provides functions concerning Brownian motion
and Langevin motion on manifolds. The following three
perspectives are available:

1. Symbolic computations
2. Simulations of sample paths
3. Estimation of the intrinsic metric tensor $g$

The symbolic computations and simulations can be done both
extrinsically and intrinsically, although the latter is usually
faster in both symbolic computations and in simulations since the
dimensions of the driving noise is smaller than it is in the extrinsic
setting. The estimations solely concern themselves with extrinsic sample paths
and recovering the intrinsic geometry from observing many paths.

## Installation

You can install the package via from github on windows in command line with:

``` 
python -m pip install git+https://github.com/shill1729/ManifoldBm.git
```

## Symbolic Computations
Here is a basic example for symbolic computations. We can
symbolically compute
1. The metric tensor
2. The orthogonal projection
3. The intrinsic and extrinsic drift coefficient
4. The intrinsic and extrinsic diffusion coefficient
```python
from ManifoldBm.ManifoldLearn import *
tn = 10**-5
C = 0.2
a = 0.01
b = 5
nens = 100
npaths = 60
ntime = 20
x0 = np.linspace(a, b, nens)
# Setting up the synthetic process
x, y, z= symbols("x y z", real=True)
F1 = x**2
f1 = F1-y
# Now we can set up an instrinsic BM
param = Matrix([x])
chart = Matrix([x, solve(f1, y)[-1]])
d = param.shape[0]
D = chart.shape[0]
dim = (D,d)
print("Extrinsic vs Intrinsic dim ="+str(dim))
print("Chart")
print(chart)
bm = IntrinsicBm(param, chart)
print(bm)
print(bm.manifold)
```
