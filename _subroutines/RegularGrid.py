import numpy as np
import matlab.engine
from scipy.interpolate import griddata

def FindBoundary(expX,grid,mat):
    """
    Find regular grid points inside geometry.

    Parameters
    ----------
    expX : (np,2),float
        Reference xy coordinates of experimental points in one surface.
    grid : (nny,nnx,2),float
        Regular grid xy coordinates.
    eng : object
        Matlab engine.

    Returns
    -------
    ptsin : (nny,nnx),bool
        Regular grid of points inside/outside geometry.

    Notes
    -----
    np : int
        Number of experimental points.
    """

    xc = mat.transpose(matlab.double(list(expX[:,0])))
    yc = mat.transpose(matlab.double(list(expX[:,1])))

    k = np.array(mat.boundary(xc,yc,0.8),int).flatten() - 1

    xcIn = matlab.double(list(np.array(xc).flatten()[k]))
    ycIn = matlab.double(list(np.array(yc).flatten()[k]))
    xxgrid = matlab.double(grid[...,0].tolist())
    yygrid = matlab.double(grid[...,1].tolist())

    ptsin = np.array(mat.inpolygon(xxgrid,yygrid,xcIn,ycIn))

    return ptsin

def RegularGrid(expX,expU,mat,nf):
    """
    Generate regular grid by interpolating experimental data.

    Parameters
    ----------
    expX : (np,3,2),float
        Reference coordinates of experimental points.
    expU : (np,3,2,nf),float
        Displacements of experimental points.
    mat : object
        Matlab engine.
    nf : int
        Number of increments.

    Returns
    -------
    grid : (nny,nnx,2,3,nf),float
        Interpolated experimental points to regular grid.

    Notes
    -----
    np : int
        Number of experimental points.
    nny : int
        Number of nodes of regular grid in y-direction.
    nnx : int
        Number of nodes of regular grid in x-direction.
    """

    # Compute pitch for regular grid
    pitchX = np.max(np.diff(np.sort(expX[:,0,:],0),1,0))
    pitchY = np.max(np.diff(np.sort(expX[:,1,:],0),1,0))

    # Compute regular grid limits
    xMin,yMin = np.min(np.nanmin(expX[:,:2,:],0),1)
    xMax,yMax = np.max(np.nanmax(expX[:,:2,:],0),1)

    # Compute number of points for regular grid
    nnx = int(round((abs(xMin) + abs(xMax))/pitchX)) + 1 
    nny = int(round((abs(yMin) + abs(yMax))/pitchY)) + 1

    # Generate xy grid coordinates
    dx = np.linspace(xMin,xMax,nnx)
    dy = np.linspace(yMax,yMin,nny)

    # Generate xy regular grid
    xx,yy,zz = np.meshgrid(dx,dy,[0,0])
    grid = np.stack((xx,yy,zz),axis=3)

    # Assign experimental points z coordinates to regular grid
    for i in range(2):
        grid[...,i,2] = griddata(expX[:,:2,i],expX[:,-1,i],grid[...,i,:2])

    # Find regular grid points inside geometry
    ptsin = np.zeros((nny,nnx,2),dtype=bool)
    for i in range(2):
        ptsin[...,i] = FindBoundary(expX[...,:2,i],grid[...,i,:],mat)

    # Assign nan to points outside geometry
    ptsin = np.logical_or(ptsin[...,0],ptsin[...,1])
    grid[~ptsin,...] = np.nan

    # Interpolate displacement field to regular grid
    gridU = np.zeros((nny,nnx,2,3,nf))
    for i in range(2):
        gridU[...,i,:,:] = griddata(expX[:,:2,i],expU[...,i,:],grid[...,i,:2])

    # Obtain deformed grid coordinates
    gridX = grid[...,None] + gridU

    # Average xy grid coordinates of both surfaces
    gridX[...,:2,:] = np.mean(gridX[...,:2,:],2)[...,None,:,:]

    # Apply savitzky-golay smoothing filter on xy plane
    # if smooth:
    #     for i in range(2):
    #         for j in range(2):
    #             surf[:,:,i,:] = savgol_filter(surf[:,:,i,:],11,2,axis=j)

    return gridX,nny,nnx