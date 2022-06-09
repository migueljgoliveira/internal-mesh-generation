import numpy as np

import _subroutines

def Volume(coord,nny,nnx,nnz):
    """
    Compute mesh elements volume by shape functions.

    Parameters
    ----------
    coord : (nny,nnx,nnz,3),float
        Coordinates of reconstructed nodes by element on regular grid.
    nny : int
        Number of nodes of regular grid in y-direction.
    nnx : int
        Number of nodes of regular grid in x-direction.
    nnz : int
        Number of nodes of regular grid in z-direction.

    Returns
    -------
    evol : (ney,nex,nez),float
        Volume of reconstructed elements on regular grid.

    Notes
    -----
    ney : int
        Number of elements of regular grid in y-direction.
    nex : int
        Number of elements of regular grid in x-direction.
    nez : int
        Number of elements of regular grid in z-direction.
    """

    # Number of elements
    ney,nex,nez = nny-1,nnx-1,nnz-1

    # Reshape coordinates by element
    coord = _subroutines.ReshapeMesh(coord,nny,nnx,nnz)

    # Compute elements volume by shape functions
    _,_,evol = _subroutines.ElHex8R(coord)

    # Reshape to matrix
    evol = np.reshape(evol,(ney,nex,nez))

    return evol