def MidSurface(gridX,gridN):
    """
    Compute middle surface between regular grid front and back surfaces.

    Parameters
    ----------
    gridX : (nny,nnx,3,3),float
        Regular grid surface coordinates.
    gridX : (nny,nnx,2,3),float
        Regular grid surface normals.
    nf : int
        Number of increments.

    Returns
    -------
    gridX : (nny,nnx,3,3),float
        Regular grid surface coordinates with middle surface.
    """

    # Extract coordinates of front surface
    surfX0 = gridX[...,0,0]
    surfY0 = gridX[...,0,1]
    surfZ0 = gridX[...,0,2]

    # Extract coordinates of back surface 
    surfX1 = gridX[...,-1,0]
    surfY1 = gridX[...,-1,1]
    surfZ1 = gridX[...,-1,2]

    # Extract surface normals of front surface
    normX0 = gridN[...,0,0]
    normY0 = gridN[...,0,1]
    normZ0 = gridN[...,0,2]

    # Extract surface normals of back surface
    normX1 = gridN[...,-1,0]
    normY1 = gridN[...,-1,1]
    normZ1 = gridN[...,-1,2]

    # Compute middle surface in z coordinates
    gridX[...,1,2] = (surfZ0 + surfZ1)/2

    # Distance of outside surfaces from middle surface in z coordinates
    surfH0 = abs(gridX[...,1,2] - surfZ1)
    surfH1 = abs(gridX[...,1,2] - surfZ1)

    # Compute middle surface in x coordinates
    midX0 = normX0*surfH0/abs(normZ0) + surfX0
    midX1 = normX1*surfH1/abs(normZ1) + surfX1
    gridX[...,1,0] = (midX0 + midX1)/2

    # Compute middle surface in y coordinates
    midY0 = normY0*surfH0/abs(normZ0) + surfY0
    midY1 = normY1*surfH1/abs(normZ1) + surfY1
    gridX[...,1,1] = (midY0 + midY1)/2

    return gridX