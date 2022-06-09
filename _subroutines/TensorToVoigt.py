import numpy as np

def TensorToVoigt(tensor,nf=None):
    """ 
    Convert strain tensor to voigt form.

    Parameters
    ----------
    tensor : (ne,3,3),float
        Strain in tensor form.
    nf : int
        Number of increments.

    Returns
    -------
    voigt : (ne,6),float
        Strain in voigt notation.

    Notes
    -----
    ne : int
        Number of elements.
    """

    if nf is None:
        ne = tensor.shape[0]
        voigt = np.zeros((ne,6))
    else:
        ne = tensor.shape[1]
        voigt = np.zeros((nf,ne,6))

    voigt[...,0] = tensor[...,0,0]
    voigt[...,1] = tensor[...,1,1]
    voigt[...,2] = tensor[...,2,2]
    voigt[...,3] = tensor[...,0,1]*2
    voigt[...,4] = tensor[...,0,2]*2
    voigt[...,5] = tensor[...,1,2]*2

    if nf is not None:
        voigt = np.moveaxis(voigt,0,-1)

    return voigt