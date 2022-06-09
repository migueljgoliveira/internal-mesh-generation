import _subroutines

def LocalVolume(f,eFEM,gridX,gridZ,nny,nnx,nnz,reference,strategy,speed):

    if reference == 'total':
        ref = 0
    elif reference == 'incremental':
        ref = f - 1

    # Through optimization of logarithmic bias
    if strategy == 'bias':
        gridZ[...,f] = _subroutines.BiasZ(gridX[...,f],gridZ[...,f],
                                          eFEM['EVOL'][...,f],
                                          eFEM['EVOL'][...,ref],
                                          nny,nnx,nnz,
                                          speed)

    # Through direct optimization of each layer
    elif strategy == 'optimise':
        gridZ[...,f] = _subroutines.OptimiseZ(gridX[...,f],gridZ[...,f],
                                              eFEM['EVOL'][...,f],
                                              eFEM['EVOL'][...,ref],
                                              nny,nnx,nnz,
                                              speed)

    # Reconstruct undeformed configuration through bezier curve
    eFEM['X'][...,f] = _subroutines.Bezier(gridX[...,f],gridZ[...,f])

    # Compute elements volume in deformed configuration
    eFEM['EVOL'][...,f] = _subroutines.Volume(eFEM['X'][...,f],
                                              nny,nnx,nnz)

    return eFEM['X'][...,f],eFEM['EVOL'][...,f],gridZ[...,f]