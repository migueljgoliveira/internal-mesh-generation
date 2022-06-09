import pickle

def ExportPickle(eFEM,dir):
    """
    Export experimental finite element mesh to pickle file.

    Parameters
    ----------
    eFEM : dict
        Final experimental finite element mesh dict.
    dir : str
        Directory of project to export output files.
    """

    outF = f'output\\{dir}\\{dir}'

    # Save experimental finite element mesh to pickle file
    with open(f'{outF}.pkl','wb') as f:
        pickle.dump(eFEM,f)

    return