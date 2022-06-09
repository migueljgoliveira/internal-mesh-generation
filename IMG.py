import warnings
import numpy as np
import matlab.engine
from functools import partial
from p_tqdm import t_map, p_map
from tqdm import tqdm

import _subroutines

warnings.filterwarnings('ignore')
# import matplotlib.pyplot as plt

def IMG(name,nnz,xlims,ylims,zlims,output,options):


    ##################
    # PRE-PROCESSING #
    ##################

    # Start matlab engine
    mat = matlab.engine.start_matlab()

    # Create output directory
    dir = _subroutines.CreateDirectory(name,output)

    # Load experimental coordinates and displacements
    expX,expU,nf = _subroutines.LoadExperimental(name,zlims)

    # Interpolate coordinates and displacements to regular grid (front/back)
    gridX,nny,nnx = _subroutines.RegularGrid(expX,expU,mat,nf)

    # Compute regular grid surface normals
    gridN = _subroutines.SurfaceNormals(gridX,mat,nny,nnx,nf)

    # Trim regular grid in x,y directions
    gridX,gridN,nny,nnx = _subroutines.TrimGrid(gridX,gridN,xlims,ylims)

    # Allocate space in regular grid for middle surface
    gridX = np.insert(gridX,1,np.zeros((nny,nnx,3,nf)),2)

    # Close matlab engine
    mat.exit()

    # Compute middle surface between front and back surfaces
    for f in range(nf):
        gridX[...,f] = _subroutines.MidSurface(gridX[...,f],gridN[...,f])

    # Generate linear z distribution
    gridZ = np.ones((nny,nnx,nnz,nf)) * np.linspace(0,1,nnz)[...,None]

    # Initialize experimental finite element mesh
    eFEM = _subroutines.ExperimentalFEM(nny,nnx,nnz,nf,dir)

    # Load numerical coordinates on regular grid (delete after development)
    try:
        numX,numU,numLE,numEVOL,numVOL = _subroutines.LoadNumerical(name,
                                                             ylims,xlims,zlims,
                                                             nny,nnx,nnz)
    except:
        pass

    # Number of elements
    ney,nex,nez = nny-1,nnx-1,nnz-1


    ##################
    # RECONSTRUCTION #
    ##################

    # Reconstruct deformed configurations with global volume correction
    if options['global']:
        processing = options['processing']
        eFEM,gridX,gridN = _subroutines.GlobalVolume(eFEM,
                                                     gridX,gridN,gridZ,
                                                     nny,nnx,nnz,nf,
                                                     processing)

    # Reconstruct deformed configuration without global volume correction
    else:
        if options['local']:
            hide = True
        else:
            hide = False

        for f in tqdm(range(nf),'Reconstruction',disable=hide):
            # Reconstruct deformed configuration through bezier curve
            eFEM['X'][...,f] = _subroutines.Bezier(gridX[...,f],gridZ[...,f])

            # Elements volume in deformed configuration
            eFEM['EVOL'][...,f] = _subroutines.Volume(eFEM['X'][...,f],
                                                      nny,nnx,nnz)

    # Reconstruct deformed configuration with local volume correction
    if options['local']:

        reference = options['reference']
        processing = options['processing']
        strategy = options['strategy']
        speed = options['speed']

        # Generate partial function of local volume
        func = partial(_subroutines.LocalVolume,eFEM=eFEM,
                                                gridX=gridX,gridZ=gridZ,
                                                nny=nny,nnx=nnx,nnz=nnz,
                                                reference=reference,
                                                strategy=strategy,
                                                speed=speed)

        # in parallel processing
        if (processing == 'parallel') and (reference == 'total'):
            rec = p_map(func,np.arange(1,nf),num_cpus=6,desc='Local Volume')

        # in sequential processing
        elif processing == 'sequential':
            rec = t_map(func,np.arange(1,nf),desc='Local Volume')

        # Extract results from reconstruction
        for f in range(1,nf):
            eFEM['X'][...,f] = rec[f-1][0]
            eFEM['EVOL'][...,f] = rec[f-1][1]
            gridZ[...,f] = rec[f-1][2]


    ###################
    # POST-PROCESSING #
    ###################

    # Displacements wrt reference configuration
    eFEM['U'] = eFEM['X'] - eFEM['X'][...,0,None]

    # Logarithmic strain wrt reference configuration
    eFEM['LE'] = _subroutines.LogStrain(eFEM['X'][...,0],eFEM['U'],
                                        nny,nnx,nnz,nf)

    # Global volume as the sum of elements volume
    eFEM['VOL'] = (np.ones((ney,nex,nez,nf))
                       * np.sum(eFEM['EVOL'],(0,1,2))[None,None,None,:])

    try:
        eFEM['rX'] = numX
        eFEM['rU'] = numU
        eFEM['rLE'] = numLE
        eFEM['rEVOL'] = numEVOL
        eFEM['rVOL'] = numVOL
    except:
        eFEM['rX'] = np.zeros((nny,nnx,nnz,3,nf))
        eFEM['rU'] = np.zeros((nny,nnx,nnz,3,nf))
        eFEM['rLE'] = np.zeros((ney,nex,nez,6,nf))
        eFEM['rEVOL'] = np.zeros((ney,nex,nez,nf))
        eFEM['rVOL'] = np.zeros((ney,nex,nez,nf))

    # Update experimental finite element mesh with mesh
    eFEM = _subroutines.GenerateFEM(eFEM,nny,nnx,nnz,nf)

    # Export experimental finite element mesh on different formats
    _subroutines.Export(eFEM,nf,dir)

    return

if __name__ == '__main__':

    options = {
                                        # True | False
                   'global': False,
                                        # True | False
                    'local': False,
                                        # bias | optimise
                 'strategy': 'optimise',
                                        # total | incremental
                'reference': 'total',
                                        # fast | slow
                    'speed': 'slow',
                                        # sequential | parallel
               'processing': 'parallel',
              }

    name = 'Plane-Strain-Mesh0-Thin'

    output = f'{name}'

    inp = np.loadtxt(f'input\\{name}\\{name}.inp',delimiter=',',usecols=(1,2))

    nnz = int(inp[0,0])
    xlims = inp[1,:]
    ylims = inp[2,:]
    zlims = inp[3,:]

    IMG(name,nnz,xlims,ylims,zlims,output,options)