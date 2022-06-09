import numpy as np
from scipy.optimize import Bounds
from scipy.signal import savgol_filter
from scipy.optimize import minimize

import _subroutines

def GenerateBiasZ(w,optiZ,nnz):

    # Compute flow and weight of z-bias
    flow = np.sign(w)
    w = abs(w)

    # Generate z-bias based on logarithmic function
    n = int((nnz+1)/2)
    bias = ((10**np.linspace(0,np.log10(w+1),n)-1)/(2*w))[1:-1]

    # Flow of z-bias from inside to outside <-:->
    if flow == -1:
        optiZ[1:n-1] = bias
        optiZ[n:-1] = np.flip(abs(bias - 0.5)) + 0.5

    # Flow of z-bias from outside to inside ->:<-
    elif flow == 1:
        optiZ[1:n-1] = np.flip(abs(bias - 0.5))
        optiZ[n:-1] = np.flip(abs(optiZ[1:n-1] - 0.5)) + 0.5

    return optiZ

def Optimisation(w,Xij,Zij,Vij,nnz):

    # Modify z-bias based on weight w
    optiZ = np.copy(Zij)
    if w[0] != 0:
        optiZ[1,1,:] = GenerateBiasZ(w[0],optiZ[1,1,:],nnz)

    # Reconstruct coordinates in deformed configuration
    optiX = _subroutines.Bezier(Xij,optiZ)

    # Compute elements volume
    optiV = _subroutines.Volume(optiX,3,3,nnz)

    # Compute individual cost function
    return np.nansum(np.max(abs(optiV - Vij),2))

def BiasZ(gridX,gridZ,recV,recVr,nny,nnx,nnz,speed):

    # Number of elements
    ney,nex,nez = nny-1,nnx-1,nnz-1
    ne = ney*nex*nez

    # Create masks with nan boundaries in xy directions
    maskX = np.full((nny+2,nnx+2,3,3),np.nan)
    maskZ = np.full((nny+2,nnx+2,nnz,2),np.nan)
    maskVr = np.full((ney+2,nex+2,nez),np.nan)

    # Assign masks with values
    maskX[1:-1,1:-1,...] = gridX
    maskZ[1:-1,1:-1,...] = gridZ[...,None]
    maskVr[1:-1,1:-1,...] = recVr

    # Compute cost function
    cost = np.nansum(np.max(abs(recV - maskVr[1:-1,1:-1,...]),2))

    # Initialize while loop control variables
    w0 = 0.0
    it = 0
    tol = 1e-8

    print(f'\nIteration: {it} ({cost/ne:.8f})')
    while True:
        for i in range(1,nny+1):
            i0,i1 = i-1,i+2
            for j in range(1,nnx+1):
                j0,j1 = j-1,j+2

                # Opimize z distribution of nodes ij
                if speed == 'slow':
                    opti = minimize( Optimisation,
                                     x0=[w0],
                                     method='Nelder-Mead',
                                     options={'adaptive': True,
                                              'xatol': 1e-8,
                                              'fatol': 1e-8},
                                     args=(maskX[i0:i1,j0:j1,...],
                                           maskZ[i0:i1,j0:j1,:,0],
                                           maskVr[i0:i1-1,j0:j1-1,:],
                                           nnz)
                                   )

                elif speed == 'fast':
                    opti = minimize( Optimisation,
                                     x0=w0,
                                     method='SLSQP',
                                     options={'fatol': 1e-8},
                                     args=(maskX[i0:i1,j0:j1,...],
                                           maskZ[i0:i1,j0:j1,:,0],
                                           maskVr[i0:i1-1,j0:j1-1,:],
                                           nnz)
                                   )

                # Check if individual solution improved
                w1 = opti.x[0]
                if w1 != w0:
                    maskZ[i,j,:,0] = GenerateBiasZ(w1,maskZ[i,j,:,0],nnz)

        # Smooth z-bias in x-direction
        for i in range(1,nny+1):
            for k in range(1,nnz-1):
                maskZ[i,1:-1,k,0] = savgol_filter(maskZ[i,1:-1,k,0],nnx,2)

        # Smooth z-bias in y-direction
        for j in range(1,nnx+1):
            for k in range(1,nnz-1):
                maskZ[1:-1,j,k,0] = savgol_filter(maskZ[1:-1,j,k,0],nny,2)

        # Reconstruct coordinates in deformed configuration
        itX = _subroutines.Bezier(gridX,maskZ[1:-1,1:-1,:,0])

        # Compute volume
        itV = _subroutines.Volume(itX,nny,nnx,nnz)

        # Compute cost function
        ncost = np.nansum(np.max(abs(itV - maskVr[1:-1,1:-1,...]),2))

        # Check if solution improved
        print(f'Iteration: {it+1} ({ncost/ne:.8f})')

        # Stop if solution does not improve
        if (cost - ncost) < 0:
            break

        # Stop if solution improved but within tolerance
        elif (cost - ncost)/ne < tol:
            maskZ[...,1] = np.copy(maskZ[...,0])
            break

        # Continue if solution improved
        else:
            maskZ[...,1] = np.copy(maskZ[...,0])
            cost = ncost
            it += 1

    return maskZ[1:-1,1:-1,:,1]