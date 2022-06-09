import numpy as np

def PolarDecomposition(dfgrd,side='left'):
    """
    Perform the polar decomposition of the deformation gradient.

    Parameters
    ----------
    dfgrd : (ne,3,3),float
        Deformation gradient.

    Returns
    -------
    strch : (ne,3,3),float
        Deformation left or right stretch tensor.

    Notes
    -----
    ne : int
        Number of elements.

    See Also
    --------
    scipy.linalg.polar : compute the polar decomposition of a single
      matrix.

    Theory
    ------
    The deformation gradient F can be written as
        F = VR or F = UR,
      where V is the left-stretch tensor and U is the right-stretch
      tensor. U is the stretch tensor on the global csys and V is the
      stress tensor on the local csys. This way, U can be said to be
      the stretch tensor free of rigid-body rotations, so that the
      relationship between V and U is given by U = R'UR.
    """

    # Singular value decomposition of deformation gradient
    W,S,Vh = np.linalg.svd(dfgrd,full_matrices=False)

    # Left stretch tensor
    if side == 'left':
        # strch = (W * S[...,None,:]) @ np.transpose(W,(0,1,3,2))
        strch = (W * S[...,None,:]) @ np.moveaxis(W,-1,-2)

    # Right stretch tensor
    elif side == 'right':
        # strch = (np.transpose(Vh,(0,2,1)) * S[:,None,:]) @ Vh
        strch = (np.moveaxis(Vh,-1,-2) * S[...,None,:]) @ Vh

    return strch