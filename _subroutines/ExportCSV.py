import numpy as np

def ExportCSV(eFEM,nf,dir):
    """
    Export experimental finite element mesh to csv files.

    Parameters
    ----------
    eFEM : dict
        Final experimental finite element mesh dict.
    nf : int
        Number of increments.
    dir : str
        Directory of project to export output files.
    """

    outF = f'output\\{dir}\\{dir}'

    # Export nodal coordinates
    path = f'{outF}_Nodes.csv'
    data = eFEM['Mesh']['Nodes']
    head = 'X;Y;Z'
    np.savetxt(path,data,header=head,fmt='%.15f',delimiter=';',comments='')

    # Export elements connectivity
    path = f'{outF}_Elements.csv'
    data = eFEM['Mesh']['Elements']
    head = ';'.join([f'Node-{i+1}' for i in range(8)])
    np.savetxt(path,data,header=head,fmt='%d',delimiter=';',comments='')

    # Export nodal displacements
    head = 'U;V;W'
    for f in range(nf):
        path = f'{outF}_U_{f}.csv'
        data = eFEM['Mesh']['U'][...,f]
        np.savetxt(path,data,header=head,fmt='%.15f',delimiter=';',comments='')

    return
