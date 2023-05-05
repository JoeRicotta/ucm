import numpy as np
from scipy import linalg
import warnings


class UCM(object):
    """
    A simple class which performs the uncontrolled manifold analysis
    on a given jacobian-element pair.

    Parameters
    ----------
    jacobian : array_like
        An (n x p) 2-dimensional np.ndarray() which defines the jacobian matrix
        linking changes in elements to changes in peformance. Specifically,

        dy = J dX

        where
          y: performance variable,
          X: elemental variable(s), and
          J: jacobian matrix.

        Depending on the context, J may be estimated from experimental data or
        known a-priori.
        
    elements : array_like
        An (m x p) 2-dimensional np.ndarray() which defines the state of the
        performance variables being studied.

    Returns
    -------
    None.

    Attributes
    ----------
    ucm_ : array_like
        A matrix of basis vectors spanning the uncontrolled manifold. Equal
        to the basis vector spanning the nullspace of the Jacobian matrix.
        As such, this represents the space where changes in elemental variables
        result in no change in performance.

    ort_ : array_like
        A matrix of basis vectors spanning the orthogonal complement to the
        Jacobian. Equivalent to the jacobian matrix itself, scaled to have
        vectors of unit length.

    dim_ucm_ : int
        The dimensionality of the uncontrolled manifold.

    dim_ort_ : int
        The dimensionality of the orthogonal space.

    onb_ : array_like
        A square projection matrix formed through the union of the basis
        vectors spanning the UCM and spanning the ORT space.

    vucm_ : float
        The variance of the elemental variables along the uncontrolled manifold.

    vort_ : float
        The variance of the elemental variables along the orthogonal manifold.

    synergy_index_ : float
        A value ranging from -1 to 1 which represents the manner in which elements
        coordinated to stabilize the performance variable.
    

    Example
    -----
    >>> import numpy as np
    >>> from ucm import UCM
    >>>
    >>> j = np.random.rand(2,3) # fake jacobian matrix
    >>> X = np.random.rand(30,3) # fake elements
    >>> ucm = UCM(j,X) # use ucm package
    >>> dir(ucm) # see available values
    
    """

    def __init__(self, jacobian, elements):
        self.jacobian_ = jacobian
        self.elements_ = elements

        # checking that jacobian is 2-dimensional
        n_rows, n_cols = jacobian.shape
        if n_rows > n_cols:
            warnings.warn(
                f"Jacobian matrix is long, with {n_rows} rows > {n_cols} columns. "
                "Expect weird behavior from an overdetermined system."
            )
        
        # making sure j is normalized
        norm = np.diag((1 / linalg.norm(self.jacobian_, axis = 1)))
        normalized_jacobian = norm @ self.jacobian_

        # getting basis of nullspace of the jacobian matrix (ucm) and dimension
        ucm = linalg.null_space(normalized_jacobian)
        dim_ucm = ucm.shape[1]

        # getting ort from ucm and dimension
        ort = linalg.null_space(ucm.T)
        dim_ort = ort.shape[1]

        # forming an orthonormal basis using UCM and ORT vectors
        onb = np.concatenate((ort, ucm), axis = 1)

        # now making projections onto orthonormal basis
        cov_x = np.cov((self.elements_ @ onb).T)
        vort = np.diag(cov_x)[:dim_ort].sum()
        vucm = np.diag(cov_x).sum() - vort
        dv = (vucm - vort) / (vucm + vort)

        # storing all values
        self.ucm_ = ucm
        self.ort_ = ort
        self.dim_ucm_ = dim_ucm
        self.dim_ort_ = dim_ort
        self.onb_ = onb
        self.vucm_ = vucm
        self.vort_ = vort
        self.synergy_index_ = dv

    def __repr__(self):
        return "UCM()"

# brief test
if __name__ == "__main__":
    j = np.random.rand(2,3)
    X = np.random.rand(30,3)
    ucm = UCM(j,X)
    dir(ucm)

