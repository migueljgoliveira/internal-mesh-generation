import numpy as np

def TrimGrid(gridX,gridN,xlims,ylims):
    """
    Trim regular grid in x and y directions.

    Parameters
    ----------
    gridX : (nny,nnx,2,3,nf),float
        Regular grid surface coordinates.
    gridN : (nny,nnx,2,3,nf),float
        Regular grid surface normals.
    xlims : (2,)
        Limits to trim regular grid in x-direction.
    ylims : (2,)
        Limits to trim regular grid in y-direction.

    Returns
    -------
    gridX : (nny,nnx,2,3,nf),float
        Trimmed regular grid surface coordinates.
    gridN : (nny,nnx,2,3,nf),float
        Trimmed regular grid surface normals.
    nny : int
        Number of nodes of trimmed regular grid in y-direction.
    nnx : int
        Number of nodes of trimmed regular grid in x-direction.
    """

    # Trim regular grid in x-direction
    nanmask = np.isnan(gridX[...,0,0,0])

    lbmask = gridX[...,0,0,0] >= xlims[0]
    ubmask = gridX[...,0,0,0] <= xlims[1]

    lbmask = np.logical_xor(nanmask,lbmask)
    ubmask = np.logical_xor(nanmask,ubmask)

    mask = np.logical_and(lbmask,ubmask)
    idxtrim = np.where(np.logical_not(mask.any(0)))[0]

    gridX = np.delete(gridX,idxtrim,1)
    gridN = np.delete(gridN,idxtrim,1)

    # Trim regular grid in y-direction
    nanmask = np.isnan(gridX[...,0,0,0])

    lbmask = gridX[...,0,1,0] >= ylims[0]
    ubmask = gridX[...,0,1,0] <= ylims[1]

    lbmask = np.logical_xor(nanmask,lbmask)
    ubmask = np.logical_xor(nanmask,ubmask)

    mask = np.logical_and(lbmask,ubmask)
    idxtrim = np.where(np.logical_not(mask.any(1)))[0]

    gridX = np.delete(gridX,idxtrim,0)
    gridN = np.delete(gridN,idxtrim,0)

    # Number of nodes in trimmed regulare grid
    nny,nnx = gridX.shape[:2]

    return gridX,gridN,nny,nnx