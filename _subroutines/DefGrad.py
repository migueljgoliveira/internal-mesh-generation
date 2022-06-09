import numpy as np

def DefGrad(displ,dNdNr,jac):
    """ 
    Compute the deformation gradient.

    Parameters
    ----------
    displ : (ne,8,3),float
        Nodal displacements.
    dNdNr : (ne,3,3),float
        Shape function derivatives wrt. natural  coordinates.
    jac : (ne,3,3),float
        Jacobian matrix.

    Returns
    -------
    dfgrd : (ne,3,3),float
        Deformation gradient.

    Notes
    -----
    ne : int
        Number of elements.
    """

    # Derivatives of shape functions wrt. cartesian coordinates
    dNdX = np.linalg.inv(jac) @ dNdNr

    # Deformation gradient
    dUdX = dNdX @ displ

    dfgrd = np.identity(3) + dUdX

    return dfgrd