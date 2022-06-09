import os
import numpy as np
from scipy.interpolate import griddata

import _subroutines

def LoadNumerical(name,ylims,xlims,zlims,nny,nnx,nnz):

    dir = f'input\\{name}\\fem'

    try:
        data = np.load(f'{dir}\\{name}_Grid.npz')
        gridX = data['arr_0']
        gridU = data['arr_1']
        gridLE = data['arr_2']
        gridEVOL = data['arr_3']
        gridVOL = data['arr_4']
        # # Extract z coordinates
        # gridZ = gridX[...,-1,:]

        # # Compute min and max of z coordinates
        # minZ = np.min(gridZ,2)
        # maxZ = np.max(gridZ,2)

        # # Compute normalised z coordinates
        # gridZ = (gridZ-gridZ[...,None,:])/(gridZ[...,None,:]-gridZ[...,None,:])

    except:
        # Get number of increments
        files = os.listdir(f'{dir}')
        nf = 0
        for file in files:
            try:
                if file.split('_')[-2] == 'U':
                    nf = nf + 1
            except:
                pass

        # Load nodal coordinates
        numX = np.loadtxt(f'{dir}\\{name}_fem_X.csv',skiprows=1,delimiter=';')

        # Translate xy origin to center of specimen
        lmin = np.nanmin(numX,0)
        lmax = np.nanmax(numX,0)
        numX = numX - (lmin + lmax)/2

        # Number of nodes
        nn = numX.shape[0]

        # Load nodal displacements
        numU = np.zeros((nn,3,nf))
        for f in range(nf):
            numU[...,f] = np.loadtxt(f'{dir}\\{name}_fem_U_{f}.csv',
                                     skiprows=1,delimiter=';')

        # Generate xy grid coordinates
        dx = np.linspace(xlims[0],xlims[1],nnx)
        dy = np.linspace(ylims[1],ylims[0],nny)
        dz = np.linspace(zlims[1],zlims[0],nnz)

        # Generate xy regular grid
        xx,yy,zz = np.meshgrid(dx,dy,dz)
        grid = np.stack((xx,yy,zz),axis=3)

        # Interpolate numerical displacements to regular grid
        gridU = griddata(numX,numU,grid)

        # Compute numerical coordinates on regular grid
        gridX = grid[...,None] + gridU

        # Compute logarithmic strain on regular grid
        gridLE = _subroutines.LogStrain(gridX[...,0],gridU,nny,nnx,nnz,nf)

        gridEVOL = np.zeros((nny-1,nnx-1,nnz-1,nf))
        for f in range(nf):
            gridEVOL[...,f] = _subroutines.Volume(gridX[...,f],nny,nnx,nnz)

        gridVOL = (np.ones((nny-1,nnx-1,nnz-1,nf))
                       * np.sum(gridEVOL,(0,1,2))[None,None,None,:])

        np.savez(f'{dir}\\{name}_Grid',gridX,gridU,gridLE,gridEVOL,gridVOL)

    return gridX,gridU,gridLE,gridEVOL,gridVOL
