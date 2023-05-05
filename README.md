# Uncontrolled manifold analysis
A simple python class to handle the uncontrolled manifold (UCM) analysis.

In the literature, the uses of the UCM analysis are myriad and the descriptions all fairly opaque. In general, the UCM analysis proceeds through the selection of i.) elemental variables, and ii.) performance variables. In principle, elemental variables are body-level variables whereas the performance variable is framed as a function of elemental variables.

In short, the UCM analyzes the composition of elemental variance aligned with the null space of the gradient between elemental and performance variables. To be more precise, define the following as

$X$ : an $m \times n$ matrix representing m observations of n elemental variables,
$Y$ : an $m \times p$ matrix representing m observations of p performance variables, and
$J$ : a $p \times n$ Jacobian matrix representing the partial derivatives of X wrt Y, where
$$dY' = J dX'$$

Using this package, the entire UCM analysis can be completed by entering the jacobian matrix $J$ and elements $X$ into an `UCM()` object,

```python
import numpy as np
from ucm import UCM

# simulating elements and jacobian matrix
X = np.random.rand(30,3)
J = np.random.rand(2,3)

# using simulated data to perform the UCM analysis
ucm = UCM(jacobian=J,elements=X)
```

Once observed, the elemental variables are projected onto an orthonormal basis $S$ defined by basis vectors spanning $\mathbf{R}^n$,

$$S = [UCM | ORT]$$

where

$UCM$ : the $n-p$ basis vectors spanning null(J), and
$ORT$ : the $p$ basis vectors spanning the orthogonal complement of the UCM.

The UCM represents the space where changes in elemental variables result in no change in performance, whereas the ORT space represents the space where changes in elemental variables result in maximal change in performance. (In another language, gradient descent algorithms use movement along the negative ORT direction to minimize some function represented by the gradient.)

Once the orthonormal basis $S$ is defined, we can then find the variance of the elemental variables along the UCM and ORT space by projecting the covariance matrix of elements onto the orthonormal basis,

$$S'C_{X}S = C_{S}$$

where

$C_{X} = \frac{1}{(M-1)}(X - \bar{X})'(X - \bar{X})$ : covariance matrix of elements, and
$C_{S}$ : covariance matrix projected onto the UCM and ORT spaces.

The corresponding elements along the diagonal of $C_{S}$ are then used to identify $V_{UCM}$ and $V_{ORT}$ which are the variances along the UCM and ORT subspaces, respectively. These values are then normalized per dimension and used to compute the synergy index $\Delta V$,

$$\Delta V = \frac{V_{UCM} - V_{ORT}}{V_{TOT}}$$

All of the above is completed in the single call to `ucm()`.

```python
# The orthonormal basis S
print(ucm.onb_)

# Variances along the UCM and ORT subspaces
print(ucm.vucm_)
print(ucm.vort_)

# The synergy index
print(ucm.synergy_index_)
```







