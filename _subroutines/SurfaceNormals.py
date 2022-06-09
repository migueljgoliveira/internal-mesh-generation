import numpy as np
import matlab.engine

def SurfaceNormals(gridX,mat,nny,nnx,nf):
    """
    Compute regular grid surface normals.

    Parameters
    ----------
    gridX : (nny,nnx,2,3,nf),float
        Regular grid surface coordinates.
    mat : object
        Matlab engine.
    nny : int
        Number of nodes of regular grid in y-direction.
    nnx : int
        Number of nodes of regular grid in x-direction.
    nf : int
        Number of increments.

    Returns
    -------
    gridN : (nny,nnx,2,3,nf),float
        Regular grid surface normals.
    """

    gridN = np.zeros((nny,nnx,2,3,nf))
    for f in range(nf):
        norm = np.moveaxis(np.zeros((nny,nnx,2,3)),[3,0,1,2],[0,1,2,3])

        # Compute surface normals of front and back surfaces
        for i in range(2):
            xx = matlab.double(gridX[...,i,0,f].tolist())
            yy = matlab.double(gridX[...,i,1,f].tolist())
            zz = matlab.double(gridX[...,i,2,f].tolist())
            norm[...,i] = np.array(mat.surfnorm(xx,yy,zz,nargout=3))

        # Make z normals of back surface point towards center of the specimen
        norm[...,-1] = -norm[...,-1]

        gridN[...,f] = np.moveaxis(norm,[0,1,2,3],[3,0,1,2])

    return gridN