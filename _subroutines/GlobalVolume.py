import numpy as np
from tqdm import tqdm
from functools import partial
from p_tqdm import t_map, p_map
from sklearn.metrics import r2_score
from scipy.optimize import least_squares

import _subroutines

def Optimisation(w,gridX,gridN,gridZ,Vr,nny,nnx,nnz):

    # Update surface normals through z component
    optiN =  [1.0,1.0,w[0]] * gridN
    mag = np.sqrt(np.sum(optiN**2,3))
    optiN = optiN / mag[...,None]

    # Middle surface between front and back surfaces
    tmpX = _subroutines.MidSurface(gridX,optiN)

    # Reconstruct deformed configuration through bezier curve
    optiX = _subroutines.Bezier(tmpX,gridZ)

    # Elements volume in deformed configuration
    optiV = _subroutines.Volume(optiX,nny,nnx,nnz)

    # Total volume
    tmpV = np.sum(optiV)

    if tmpV < Vr:
        return np.ones_like(optiV).flatten()*1e7
    else:
        return optiV.flatten()

def GlobalCorrection(f,eFEM,gridX,gridN,gridZ,pfit,nny,nnx,nnz):

    # Volume of reference by linear regression
    Vr = pfit[0]*f + pfit[-1]

    # Opimize surface normals components
    # w0 = [1.0,1.0,1.0]
    w0 = 1.0
    opti = least_squares( Optimisation,
                          x0=w0,
                          method='lm',
                          x_scale='jac',
                          args=(gridX[...,f],gridN[...,f],gridZ[...,f],
                                Vr,nny,nnx,nnz))

    # If solution improves, update configuration
    w1 = opti.x[0]
    if (w1 != w0).any():
        gridN[...,f] = [1.0,1.0,w1] * gridN[...,f]
        mag = np.sqrt(np.sum(gridN[...,f]**2,3))
        gridN[...,f] = gridN[...,f] / mag[...,None]

        # Middle surface between front and back surfaces
        gridX[...,f] = _subroutines.MidSurface(gridX[...,f],
                                               gridN[...,f])

        # Reconstructed points through bezier curve
        eFEM['X'][...,f] = _subroutines.Bezier(gridX[...,f],
                                               gridZ[...,f])

        # Elements volume in deformed configuration
        eFEM['EVOL'][...,f] = _subroutines.Volume(eFEM['X'][...,f],
                                                  nny,nnx,nnz)

    return eFEM['X'][...,f],eFEM['EVOL'][...,f],gridX[...,f],gridN[...,f]

def GlobalVolume(eFEM,gridX,gridN,gridZ,nny,nnx,nnz,nf,processing):

    # Reconstruct undeformed configuration through bezier curve
    eFEM['X'][...,0] = _subroutines.Bezier(gridX[...,0],gridZ[...,0])

    # Elements volume in undeformed configuration
    eFEM['EVOL'][...,0] = _subroutines.Volume(eFEM['X'][...,0],
                                              nny,nnx,nnz)

    elastic = True
    correlation = False
    correction = False

    for f in tqdm(range(1,nf),leave=False,desc='Global Volume'):

        # Reconstruct deformed configuration through bezier curve
        eFEM['X'][...,f] = _subroutines.Bezier(gridX[...,f],gridZ[...,f])

        # Elements volume in deformed configuration
        eFEM['EVOL'][...,f] = _subroutines.Volume(eFEM['X'][...,f],
                                                  nny,nnx,nnz)

        # Check when total volume stops evolving by elastic deformation
        if elastic:
            incs = np.arange(0,f+1)

            vol0 = np.sum(eFEM['EVOL'][...,:f+1],(0,1,2))
            vol1 = np.polyval(np.polyfit(incs,vol0,2),incs)

            r2 = r2_score(vol1,vol0)
            if r2 < 0.99:
                elasf = f
                elastic = False

        # Check when total volume stops evolving approximately linearly
        else:
            incs = np.arange(elasf,f+1)

            if len(incs) <= 5:
                continue

            vol0 = np.sum(eFEM['EVOL'][...,elasf:f+1],(0,1,2))
            vol1 = np.polyval(np.polyfit(incs,vol0,1),incs)

            r2 = r2_score(vol1,vol0)

            if r2 >= 0.99:
                correlation = True

            if correlation and r2 < 0.99:
                incs = np.arange(elasf,f)

                vol0 = np.sum(eFEM['EVOL'][...,elasf:f],(0,1,2))
                pfit = np.polyfit(incs,vol0,1)

                i = 1
                for vol in vol0[::-1]:
                    if np.polyval(pfit,f-i) > vol:
                        linf = f - i
                        break
                    i += 1

                correction = True
                break

    if correction:
        # Generate partial function of local volume
        func = partial(GlobalCorrection,eFEM=eFEM,
                                        gridX=gridX,gridZ=gridZ,gridN=gridN,
                                        pfit=pfit,
                                        nny=nny,nnx=nnx,nnz=nnz)

        # in parallel processing
        if processing == 'parallel':
            rec = p_map(func,np.arange(linf,nf),num_cpus=6,
                        desc='Global Volume')

        # in sequential processing
        elif processing == 'sequential':
            rec = t_map(func,np.arange(linf,nf),desc='Global Volume')

        # Extract results from reconstruction
        for f in range(linf,nf):
            i = f - linf
            eFEM['X'][...,f] = rec[i][0]
            eFEM['EVOL'][...,f] = rec[i][1]
            gridX[...,f] = rec[i][2]
            gridN[...,f] = rec[i][3]

    return eFEM,gridX,gridN