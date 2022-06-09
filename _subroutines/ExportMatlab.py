import scipy.io

def ExportMatlab(eFEM,dir):
    """
    Export experimental finite element mesh to mat file.

    Parameters
    ----------
    eFEM : dict
        Final experimental finite element mesh dict.
    dir : str
        Directory of project to export output files.
    """

    outF = f'output\\{dir}\\{dir}'

    # Save experimental finite element mesh to mat file
    with open(f'{outF}.mat','wb') as f:
        scipy.io.savemat(f,eFEM)

    return