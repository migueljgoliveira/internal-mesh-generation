import os
import numpy as np

def LoadExperimental(name,zlims):
    """
    Load experimental coordinates and displacements of two surfaces.

    Parameters
    ----------
    name : str
        Name of current project.
    zlims : (2,)
        Limits of regular grid in z-direction.

    Returns
    -------
    expX : (np,3,2),float
        Reference coordinates of experimental points.
    expU : (np,3,2,nf),float
        Displacements of experimental points.
    nf : int
        Number of increments.

    Notes
    -----
    np : int
        Number of experimental points.
    """

    expX,expU = [],[]
    for surf in ['front','back']:
        dir = f'input\\{name}\\{surf}'
        fpref = f'{dir}\\{name}_{surf}'

        # Load coordinates
        file = f'{fpref}_X.csv'
        expXsurf = np.loadtxt(file,skiprows=1,delimiter=';')
        pts = expXsurf.shape[0]

        expX.append(expXsurf)

        # Load displacements
        nf = len(os.listdir(f'{dir}')) - 1
        expUsurf = np.zeros((pts,3,nf))

        for i in range(nf):
            file = f'{fpref}_U_{i}.csv'
            expUsurf[...,i] = np.loadtxt(file,skiprows=1,delimiter=';')

        expU.append(expUsurf)

    expX = np.moveaxis(np.array(expX),[1,2,0],[0,1,2])
    expU = np.moveaxis(np.array(expU),[1,2,0,3],[0,1,2,3])

    # Verify if all points have coordinates and displacements

    # Translate undeformed coordinates to z-plane
    expX[:,2,:] = expX[:,2,:] - zlims

    # Translate xy origin to center of specimen
    lmin = np.nanmin(expX[:,:2,:2],0)
    lmax = np.nanmax(expX[:,:2,:2],0)
    expX[:,:2,:2] = expX[:,:2,:2] - (lmin + lmax)/2

    # Flip back surface z displacements
    expU[:,2,1,:] = -expU[:,2,1,:]

    return expX,expU,nf