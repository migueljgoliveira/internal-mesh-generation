import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import Bounds, minimize, differential_evolution


import _subroutines

def Optimisation(w,Xij,Zij,Vij,nnz):

    # Update z distribution of node ij
    optiZ = np.copy(Zij)
    optiZ[1,1,1:int(nnz/2)] = w
    optiZ[1,1,int(nnz/2)+1:-1] = 1 - np.flip(w)

    # Reconstruct coordinates in deformed configuration
    optiX = _subroutines.Bezier(Xij,optiZ)

    # Compute elements volume
    optiV = _subroutines.Volume(optiX,3,3,nnz)

    # Compute individual cost function
    return np.nansum(np.max(abs(optiV - Vij),2))

def OptimiseZ(gridX,gridZ,recV,recVr,nny,nnx,nnz,speed):

    # Number of elements
    ney,nex,nez = nny-1,nnx-1, nnz-1
    ne = ney*nex*nez

    # Create masks with nan boundaries in xy directions
    maskX = np.full((nny+2,nnx+2,3,3), np.nan)
    maskZ = np.full((nny+2,nnx+2,nnz,2), np.nan)
    maskVr = np.full((ney+2,nex+2,nez), np.nan)

    # Assign masks with values
    maskX[1:-1,1:-1,...] = gridX
    maskZ[1:-1,1:-1,...] = gridZ[...,None]
    maskVr[1:-1,1:-1,...] = recVr

    # Compute cost function
    cost = np.nansum(np.max(abs(recV - maskVr[1:-1,1:-1,...]),2))

    # Initial linear solution and bounds
    dw = 0.5/int(nnz-1)
    w0 = np.linspace(0,0.5,int((nnz+1)/2))[1:-1]
    bounds = Bounds(w0 - dw,w0 + dw)

    # Initialize while loop control variables
    it = 0
    tol = 1e-8

    # print(f'\nIteration: {it} ({cost/ne:.8f})')
    while True:
        for i in range(1,nny + 1):
            i0,i1 = i-1,i+2
            for j in range(1,nnx + 1):
                j0,j1 = j-1,j+2

                if speed == 'slow':
                    opti = differential_evolution( Optimisation,
                                            x0=w0,
                                            bounds=bounds,
                                            mutation=0.5,
                                            args=(maskX[i0:i1,j0:j1,...],
                                                  maskZ[i0:i1,j0:j1,:,0],
                                                  maskVr[i0:i1-1,j0:j1-1,:],
                                                  nnz))

                elif speed == 'fast':
                    opti = minimize( Optimisation,
                                     x0=w0,
                                     bounds=bounds,
                                     method='SLSQP',
                                     options={'fatol': 1e-8},
                                     args=(maskX[i0:i1,j0:j1,...],
                                           maskZ[i0:i1,j0:j1,:,0],
                                           maskVr[i0:i1-1,j0:j1-1,:],
                                           nnz)
                                    )

                # Check if individual solution improved
                w1 = opti.x
                if (w0 != w1).any():
                    maskZ[i,j,1:int(nnz/2),0] = w1
                    maskZ[i,j,int(nnz/2)+1:-1,0] = 1 - np.flip(w1)

        # Smooth z-bias in x-direction
        for i in range(1,nny + 1):
            for k in range(1,nnz - 1):
                maskZ[i,1:-1,k,0] = savgol_filter(maskZ[i,1:-1,k,0],nnx,2)

        # Smooth z-bias in y-direction
        for j in range(1,nnx + 1):
            for k in range(1,nnz - 1):
                maskZ[1:-1,j,k,0] = savgol_filter(maskZ[1:-1,j,k,0],nny,2)

        # Reconstruct coordinates in deformed configuration
        itX = _subroutines.Bezier(gridX,maskZ[1:-1,1:-1,:,0])

        # Compute volume
        itV = _subroutines.Volume(itX,nny,nnx,nnz)

        # Compute cost function
        ncost = np.nansum(np.max(abs(itV - maskVr[1:-1,1:-1,...]), 2))

        # Check if solution improved
        # print(f'Iteration: {it+1} ({ncost/ne:.8f})')

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