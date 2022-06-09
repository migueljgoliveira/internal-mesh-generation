import os
import shutil
import meshio

def ExportParaview(eFEM,nf,dir):
    """
    Export experimental finite element mesh to paraview file.

    Parameters
    ----------
    eFEM : dict
        Final experimental finite element mesh dict.
    nf : int
        Number of increments.
    dir : str
        Directory of project to export output files.
    """

    # Nodes and elements connectivity
    points = eFEM['Mesh']['Nodes']
    cells = [('hexahedron',eFEM['Mesh']['Elements'])]

    # Swap out-of-plane shear strain components for paraview notation
    eFEM['Mesh']['LE'] = eFEM['Mesh']['LE'][:,[0,1,2,3,5,4],:]
    eFEM['Mesh']['rLE'] = eFEM['Mesh']['rLE'][:,[0,1,2,3,5,4],:]

    # Output paraview file
    outF = f'{os.getcwd()}\\output\\{dir}\\{dir}'
    with meshio.xdmf.TimeSeriesWriter(f'{outF}.xdmf') as w:
        w.write_points_cells(points,cells)
        for f in range(nf):
            w.write_data(f,
                         point_data = {   'X': eFEM['Mesh']['X'][...,f],
                                          'U': eFEM['Mesh']['U'][...,f],
                                         'rX': eFEM['Mesh']['rX'][...,f],
                                         'rU': eFEM['Mesh']['rU'][...,f],
                                      },
                         cell_data  = {  'LE': [eFEM['Mesh']['LE'][...,f]],
                                       'EVOL': [eFEM['Mesh']['EVOL'][...,f]],
                                        'VOL': [eFEM['Mesh']['VOL'][...,f]],
                                        'rLE': [eFEM['Mesh']['rLE'][...,f]],
                                      'rEVOL': [eFEM['Mesh']['rEVOL'][...,f]],
                                       'rVOL': [eFEM['Mesh']['rVOL'][...,f]],
                                      })

    # Move .h5 file from cwd to output folder
    shutil.move(f'{dir}.h5',f'{outF}.h5')

    return