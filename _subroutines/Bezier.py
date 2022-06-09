import numpy as np

def Bezier(gridX,gridZ):
    """
    Reconstruct internal points through internal mesh generation method.

    Parameters
    ----------
    gridX : (nny,nnx,3,3),float
        Regular grid surface coordinates.
    gridZ : (nny,nnx,nnz),float
        Z-position of points along Bézier curves.

    Returns
    -------
    recX : (nny,nnx,nnz,3),float
        Reconstructed points on regular grid.

    Notes
    -----
    nny : int
        Number of nodes of regular grid in y-direction.
    nnx : int
        Number of nodes of regular grid in x-direction.
    nnz : int
        Number of nodes of regular grid in z-direction.
    """

    # Extract coordinates of front surface
    surf0 = gridX[..., 0,:][...,None,:]
    surfM = gridX[..., 1,:][...,None,:]
    surf1 = gridX[...,-1,:][...,None,:]

    # Repeat z-position for all coordinates components
    z = np.repeat(gridZ[...,None],3,3)

    # Reconstruct internal points
    recX = (1-z)**2*surf0 + 2*(1-z)*z*surfM + z**2*surf1

    return recX