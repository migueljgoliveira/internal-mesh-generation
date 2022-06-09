import numpy as np

def ExperimentalFEM(nny,nnx,nnz,nf,dir):
    """
    Initialize experimental finite element mesh dict for storage.

    Parameters
    ----------
    nny : int
        Number of nodes of regular grid in y-direction.
    nnx : int
        Number of nodes of regular grid in x-direction.
    nnz : int
        Number of nodes of regular grid in z-direction.
    nf : int
        Number of increments.
    dir : str
        Directory of project to export output files.

    Returns
    -------
    eFEM : dict
        Empty experimental finite element mesh dict.
    """

    # Number of elements
    ney,nex,nez = nny-1,nnx-1,nnz-1

    # Initialize experimental finite element mesh
    eFEM = {   'Name': f'{dir}',
               'X': np.zeros((nny,nnx,nnz,3,nf)),
            #    'U': np.zeros((nny,nnx,nnz,3,nf)),
            #   'LE': np.zeros((ney,nex,nez,6,nf)),
            'EVOL': np.zeros((ney,nex,nez,  nf)),
            #  'VOL':  np.ones((ney,nex,nez,  nf)),
            }

    return eFEM