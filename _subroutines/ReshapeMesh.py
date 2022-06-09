import numpy as np

def ReshapeMesh(fieldN,nny,nnx,nnz,nf=None):
    """
    Reshape mesh field by elements.

    Parameters
    ----------
    fieldN : (nny,nnx,nnz,3) or (nny,nnx,nnz,3,nf),float
        Field of reconstructed nodes on regular grid.
    nny : int
        Number of nodes of regular grid in y-direction.
    nnx : int
        Number of nodes of regular grid in x-direction.
    nnz : int
        Number of nodes of regular grid in z-direction.
    nf : int
        Number of increments.

    Returns
    -------
    fieldE : (ne,8,3) or (nf,ne,8,3),float
        Field of reconstructed nodes by element on regular grid.

    Notes
    -----
    ne : int
        Number of elements of regular grid.
    """

    # Number of elements
    ney,nex,nez = nny-1,nnx-1,nnz-1

    # Reshape mesh coordinates by element
    x = np.array([1,1,1,1,0,0,0,0])
    y = np.array([0,1,1,0,0,1,1,0])
    z = np.array([0,0,1,1,0,0,1,1])

    if nf is None:
        fieldE = np.zeros((ney,nex,nez,8,3))
    else:
        fieldE = np.zeros((ney,nex,nez,8,3,nf))

    for i in range(ney):
        for j in range(nex):
            for k in range(nez):
                fieldE[i,j,k,...] = fieldN[i+x,j+y,k+z,...]

    # Reshape to vector
    if nf is None:
        fieldE = np.reshape(fieldE,(ney*nex*nez,8,3))
    else:
        fieldE = np.moveaxis(np.reshape(fieldE,(ney*nex*nez,8,3,nf)),-1,0)

    return fieldE