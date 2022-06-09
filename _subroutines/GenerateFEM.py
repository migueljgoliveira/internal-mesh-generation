import numpy as np

def GenerateFEM(eFEM,nny,nnx,nnz,nf):
    """
    Generate finite element mesh from regular grid.

    Parameters
    ----------
    eFEM : dict
        Experimental finite element mesh in matrix form.
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
    eFEM : dict
        Updated experimental finite element mesh with vector form.
    """

    # Number of elements
    ney,nex,nez = nny-1,nnx-1,nnz-1

    nn = nny*nnx*nnz
    ne = ney*nex*nez

    eFEM['Mesh'] = {}

    grid3D = eFEM['X'][...,0]

    # Reshape quantities from matrix to vector
    meshX = np.reshape(eFEM['X'],(nn,3,nf))
    meshU = np.reshape(eFEM['U'],(nn,3,nf))
    meshLE = np.reshape(eFEM['LE'],(ne,6,nf))
    meshEVOL = np.reshape(eFEM['EVOL'],(ne,nf))
    meshVOL = np.reshape(eFEM['VOL'],(ne,nf))

    rX = np.reshape(eFEM['rX'],(nn,3,nf))
    rU = np.reshape(eFEM['rU'],(nn,3,nf))
    rLE = np.reshape(eFEM['rLE'],(ne,6,nf))
    rEVOL = np.reshape(eFEM['rEVOL'],(ne,nf))
    rVOL = np.reshape(eFEM['rVOL'],(ne,nf))

    # Delete nan values
    eFEM['Mesh']['X'] = np.delete(meshX,np.isnan(meshX[:,0,0]),0)
    eFEM['Mesh']['U'] = np.delete(meshU,np.isnan(meshU[:,0,0]),0)
    eFEM['Mesh']['LE'] = np.delete(meshLE,np.isnan(meshLE[:,0,0]),0)
    eFEM['Mesh']['EVOL'] = np.delete(meshEVOL,np.isnan(meshEVOL[:,0]),0)
    eFEM['Mesh']['VOL'] = np.delete(meshVOL,np.isnan(meshVOL[:,0]),0)

    eFEM['Mesh']['rX'] = np.delete(rX,np.isnan(rX[:,0,0]),0)
    eFEM['Mesh']['rU'] = np.delete(rU,np.isnan(rU[:,0,0]),0)
    eFEM['Mesh']['rLE'] = np.delete(rLE,np.isnan(rLE[:,0,0]),0)
    eFEM['Mesh']['rEVOL'] = np.delete(rEVOL,np.isnan(rEVOL[:,0]),0)
    eFEM['Mesh']['rVOL'] = np.delete(rVOL,np.isnan(rVOL[:,0]),0)

    # Extract mesh coordinates
    eFEM['Mesh']['Nodes'] = eFEM['Mesh']['X'][:,:,0]

    # Generate nodes numbering
    nlbl = np.full((nny,nnx,nnz), np.nan)
    nni = 0
    for i in range(nny):
        for j in range(nnx):
            for k in range(nnz):
                if not np.isnan(grid3D[i,j,k,0]):
                    nlbl[i,j,k] = nni
                    nni = nni + 1

    # Generate elements connectivity
    x = np.array([1,1,1,1,0,0,0,0])
    y = np.array([0,1,1,0,0,1,1,0])
    z = np.array([0,0,1,1,0,0,1,1])

    meshC = np.full((ne,8), np.nan)
    nei = 0
    for i in range(ney):
        for j in range(nex):
            for k in range(nez):
                meshC[nei,:] = nlbl[i+x,j+y,k+z]
                nei = nei + 1

    nans = np.any(np.isnan(meshC),1)
    eFEM['Mesh']['Elements'] = np.array(np.delete(meshC,nans,0),int)

    return eFEM