import numpy as np

import _subroutines

def LogStrain(coord,displ,nny,nnx,nnz,nf=None):
    """
    Compute the logarithmic strain in global csys by the polar 
      decomposition of the deformation gradient.

    Parameters
    ----------
    coord : (ne,8,3),float
        Reference nodal coordinates.
    displ : (ne,8,3),float
        Nodal displacements in deformed configuration.
    nny : int
        Number of nodes of regular grid in y-direction.
    nnx : int
        Number of nodes of regular grid in x-direction.
    nnz : int
        Number of nodes of regular grid in z-direction.

    Returns
    -------
    strain : (ney,nex,nez,6),float
        Logarithmic strain in global csys.
    evol : (ney,nex,nez),float
        Elements volume.

    Notes
    -----
    ne : int
        Number of elements.
    """

    # Number of elements
    ney,nex,nez = nny-1,nnx-1,nnz-1

    # Reshape reference coordinates by element
    coord = _subroutines.ReshapeMesh(coord,nny,nnx,nnz)

    # Reshape displacements by element
    displ = _subroutines.ReshapeMesh(displ,nny,nnx,nnz,nf)

    # Derivatives of shape function, inverse of jacobian and elements volume
    dNdNr,jac,_ = _subroutines.ElHex8R(coord)

    # Deformaton gradient
    dfgrd = _subroutines.DefGrad(displ,dNdNr,jac)

    # Polar decomposition of deformation gradient
    strch = _subroutines.PolarDecomposition(dfgrd,side='left')

    # Logarithmic strain in global csys
    eigv,eigpr = np.linalg.eig(strch)
    strain = eigpr * np.log(eigv[...,None,:]) @ np.linalg.inv(eigpr)

    # Convert strain tensor to voigt notation
    strain = _subroutines.TensorToVoigt(strain,nf)

    # Reshape strain to regular grid
    if nf is None:
        strain = np.reshape(strain,(ney,nex,nez,6))
    else:
        strain = np.reshape(strain,(ney,nex,nez,6,nf))

    return strain