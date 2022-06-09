import _subroutines

def Export(eFEM,nf,dir):
    """
    Export experimental finite element mesh.

    Parameters
    ----------
    eFEM : dict
        Final experimental finite element mesh dict.
    nf : int
        Number of increments.
    dir : str
        Directory of project to export output files.
    """

    # Export experimental finite element mesh to paraview file
    _subroutines.ExportParaview(eFEM,nf,dir)

    # # Export experimental finite element mesh to mat file
    _subroutines.ExportMatlab(eFEM,dir)

    # # Export experimental finite element mesh to pickle file
    _subroutines.ExportPickle(eFEM,dir)

    # # Report experimental finite element mesh to csv files
    _subroutines.ExportCSV(eFEM,nf,dir)

    return