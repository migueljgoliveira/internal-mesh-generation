import os
import shutil

def CreateDirectory(name,output):
    """
    Create directory to export output files of current project.

    Parameters
    ----------
    name : str
        Name of current project.

    Returns
    -------
    dir : str
        Directory of project to export output files.
    """

    cwd = os.getcwd()
    dir = f'{output}'

    path = f'{cwd}\\output\\{dir}'
    if os.path.isdir(path):
        shutil.rmtree(path)

    os.mkdir(path)

    return dir

