'''
 # @ Author: hdandin
 # @ Created on: 2024-06-24 14:24:42
 # @ Modified time: 2024-07-24 17:26:28
 '''

from pathlib import Path
import numpy as np
import gmsh

def square(length=1., h=0.5):
    """ square mesh for testing fracture analysis """
    modelname = 'square_'+str(h)
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(0, 0, 0, h, 1)
    gmsh.model.geo.addPoint(length, 0, 0, h, 2)
    gmsh.model.geo.addPoint(length, length, 0, h, 3)
    gmsh.model.geo.addPoint(0, length, 0, h, 4)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addLine(2, 4, 5)
    gmsh.model.geo.addCurveLoop([1, 5, 4], 1)
    gmsh.model.geo.addCurveLoop([2, 3, -5], 2)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.synchronize()
    physdict = dict()
    physdict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [1, 2], -1, name='Omega'))
    physdict['contour_area'] = (2, gmsh.model.addPhysicalGroup(2, [1], -1, name='contour_area'))
    physdict['integration_domain'] = (2, gmsh.model.addPhysicalGroup(2, [1], -1, 
                                                                     name='integration_domain'))
    physdict['contour'] = (1, gmsh.model.addPhysicalGroup(1, [5], -1, name='contour'))
    physdict['bottom'] = (1, gmsh.model.addPhysicalGroup(1, [1], -1, name='bottom'))
    physdict['right'] = (1, gmsh.model.addPhysicalGroup(1, [2], -1, name='right'))
    physdict['top'] = (1, gmsh.model.addPhysicalGroup(1, [3], -1, name='top'))
    physdict['left'] = (1, gmsh.model.addPhysicalGroup(1, [4], -1, name='left'))
    physdict['corner'] = (0, gmsh.model.addPhysicalGroup(0, [1], -1, name='corner'))
    gmsh.model.mesh.generate()
    return modelname, physdict

def mesh_with_crack(width=100., height=100., a=50., r=10., h1=1., h2=None, translate=(-50,0.),
                    rotate=0., modelname=None):
    """ generate a gmsh mesh model of a rectangle containing
    an horizontal crack from left to rectangle center
     
    Takes:
        width, height: mesh geometry
        a: crack length
        r: distance radius of contour from crack tip
        h1, h2: element densities
        translate: translate mesh (dx,dy)
        rotate: rotate mesh (angle)
     
    Returns:
        modelname
        phys_dict: physical groups
    
    """
    if h2 is None:
        h2 = h1
    if modelname is None:
        modelname = 'mesh_with_crack_'+str(h1)
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(0, 1e-5, 0, h2, 1)
    gmsh.model.geo.addPoint(a, 0, 0, h1, 2) # crack_tip
    gmsh.model.geo.addPoint(width, 0, 0, h2, 3)
    gmsh.model.geo.addPoint(width, height/2, 0, h2, 4)
    gmsh.model.geo.addPoint(0, height/2, 0, h2, 5)
    gmsh.model.geo.addPoint(a-r, 1e-5, 0, h1, 6)
    gmsh.model.geo.addPoint(a+r, 0, 0, h1, 7)
    gmsh.model.geo.addPoint(a, r, 0, h1, 8)

    gmsh.model.geo.addLine(1, 6, 1)
    gmsh.model.geo.addLine(6, 2, 6)
    gmsh.model.geo.addLine(2, 7, 7)
    gmsh.model.geo.addLine(7, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 1, 5)
    gmsh.model.geo.addCircleArc(7, 2, 8, 8)
    gmsh.model.geo.addCircleArc(8, 2, 6, 9)

    gmsh.model.geo.addCurveLoop([1, -9, -8, 2, 3, 4, 5], 1)
    gmsh.model.geo.addCurveLoop([8, 9, 6, 7], 2)

    gmsh.model.geo.addPlaneSurface([1], 1) # Omega \ contour_area
    gmsh.model.geo.addPlaneSurface([2], 2) # contour_area

    dimtags = [(2,2),(2,1)]
    gmsh.model.geo.symmetrize(gmsh.model.geo.copy(dimtags), 0., 1., 0., 0.)
    
    gmsh.model.geo.synchronize()
    
    entities_d2 = gmsh.model.getEntities(2)
    entities_d2_contour = [(2,2),(2,10)]
    entities_d1_contour = [(1,8),(1,9),(1,11),(1,12)]
    entities_d1_left = [(1,5),(1,22)]
    entities_d1_right = [(1,3),(1,20)]
    entities_d1_bottom = [(1,21)]
    entities_d1_crack = [(1,1),(1, 6),(1,13),(1,16)]

    gmsh.model.geo.translate(entities_d2, dx=translate[0], dy=translate[1], dz=0.)
    gmsh.model.geo.rotate(entities_d2, x=0, y=0, z=0., ax=0, ay=0, az=1, angle=rotate)
    
    gmsh.model.geo.synchronize()

    physdict = {}
    physdict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [i[1] for i in entities_d2], -1, 
                                                        name='Omega'))
    physdict['contour_area'] = (2, gmsh.model.addPhysicalGroup(2, 
                                                               [i[1] for i in entities_d2_contour],
                                                               -1, name='contour_area'))
    physdict['contour'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_contour],
                                                          -1, name='contour'))
    physdict['crack'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_crack], 
                                                        -1, name='crack'))
    physdict['crack_plane'] = (1, gmsh.model.addPhysicalGroup(1, [2, 7], -1, name='crack_plane'))
    physdict['bottom'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_bottom], -1, 
                                                         name='bottom'))
    physdict['right'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_right], -1, 
                                                        name='right'))
    physdict['top'] = (1, gmsh.model.addPhysicalGroup(1, [4], -1, name='top'))
    physdict['left'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_left], -1, 
                                                       name='left'))
    physdict['crack_tip'] = (0, gmsh.model.addPhysicalGroup(0, [2], -1, name='crack_tip'))
    
    gmsh.model.mesh.generate()
    elemtags = np.append(gmsh.model.mesh.getElements(2, 10)[1][0],
                         gmsh.model.mesh.getElements(2, 15)[1][0])
    gmsh.model.mesh.reverseElements(elemtags)

    return modelname, physdict

def mesh_with_crack_edi_Xtreme(width=100., height=100., a=50., r=10., h1=1., h2=None, 
                               translate=(-50,0.), rotate=0., modelname=None):
    """ generate a gmsh mesh model of a rectangle containing
    an horizontal crack from left to rectangle center
     
    Takes:
        width, height: mesh geometry
        a: crack length
        r: distance radius of contour from crack tip
        h1, h2: element densities
        translate: translate mesh (dx,dy)
        rotate: rotate mesh (angle)
     
    Returns:
        modelname
        phys_dict: physical groups
    
    """
    if h2 is None:
        h2 = h1
    if modelname is None:
        modelname = 'mesh_with_crack_edi_Xtreme_'+str(h1)
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(0, 1e-5, 0, h2, 1)
    gmsh.model.geo.addPoint(a, 0, 0, h1, 2) # crack_tip
    gmsh.model.geo.addPoint(width, 0, 0, h2, 3)
    gmsh.model.geo.addPoint(width, height/2, 0, h2, 4)
    gmsh.model.geo.addPoint(0, height/2, 0, h2, 5)
    gmsh.model.geo.addPoint(a-r, 1e-5, 0, h1, 6)
    gmsh.model.geo.addPoint(a+r, 0, 0, h1, 7)
    gmsh.model.geo.addPoint(a, r, 0, h1, 8)

    gmsh.model.geo.addLine(1, 6, 1)
    gmsh.model.geo.addLine(6, 2, 6)
    gmsh.model.geo.addLine(2, 7, 7)
    gmsh.model.geo.addLine(7, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 1, 5)
    gmsh.model.geo.addCircleArc(7, 2, 8, 8)
    gmsh.model.geo.addCircleArc(8, 2, 6, 9)

    gmsh.model.geo.addCurveLoop([1, -9, -8, 2, 3, 4, 5], 1)
    gmsh.model.geo.addCurveLoop([8, 9, 6, 7], 2)

    gmsh.model.geo.addPlaneSurface([1], 1) # Omega \ contour_area
    gmsh.model.geo.addPlaneSurface([2], 2) # contour_area

    dimtags = [(2,2),(2,1)]
    gmsh.model.geo.symmetrize(gmsh.model.geo.copy(dimtags), 0., 1., 0., 0.)
    
    gmsh.model.geo.synchronize()
    
    entities_d2 = gmsh.model.getEntities(2)
    entities_d2_contour = [(2,2),(2,10)]
    entities_d1_contour = [(1,8),(1,9),(1,11),(1,12)]
    entities_d1_left = [(1,5),(1,22)]
    entities_d1_right = [(1,3),(1,20)]
    entities_d1_bottom = [(1,21)]
    entities_d1_crack = [(1,1),(1, 6),(1,13),(1,16)]

    gmsh.model.geo.translate(entities_d2, dx=translate[0], dy=translate[1], dz=0.)
    gmsh.model.geo.rotate(entities_d2, x=0, y=0, z=0., ax=0, ay=0, az=1, angle=rotate)
    
    gmsh.model.geo.synchronize()

    physdict = {}
    physdict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [i[1] for i in entities_d2], -1, 
                                                        name='Omega'))
    physdict['integration_domain'] = (2, gmsh.model.addPhysicalGroup(2, 
                                                               [i[1] for i in entities_d2_contour],
                                                               -1, name='integration_domain'))
    physdict['integration_domaincontour'] = (1, gmsh.model.addPhysicalGroup(1, 
                                                               [i[1] for i in entities_d1_contour],
                                                          -1, name='integration_domaincontour'))
    physdict['crack'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_crack], 
                                                        -1, name='crack'))
    physdict['crack_plane'] = (1, gmsh.model.addPhysicalGroup(1, [2, 7], -1, name='crack_plane'))
    physdict['bottom'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_bottom], -1, 
                                                         name='bottom'))
    physdict['right'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_right], -1, 
                                                        name='right'))
    physdict['top'] = (1, gmsh.model.addPhysicalGroup(1, [4], -1, name='top'))
    physdict['left'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_left], -1, 
                                                       name='left'))
    physdict['crack_tip'] = (0, gmsh.model.addPhysicalGroup(0, [2], -1, name='crack_tip'))
    
    gmsh.model.mesh.generate()
    elemtags = np.append(gmsh.model.mesh.getElements(2, 10)[1][0],
                         gmsh.model.mesh.getElements(2, 15)[1][0])
    gmsh.model.mesh.reverseElements(elemtags)

    return modelname, physdict

def mesh_with_crack_edi(width=100., height=100., a=50., r1=10., r2=20., h1=1., h2=None,
                        translate=(-50,0.), rotate=0., modelname=None):
    """ generate a gmsh mesh model of a rectangle containing
    an horizontal crack from left to rectangle center
     
    Takes:
        width, height: mesh geometry
        a: crack length
        r1, r2: distance radii of contour from crack tip
        h1, h2: element densities
        translate: translate mesh (dx,dy)
        rotate: rotate mesh (angle)
     
    Returns:
        modelname
        phys_dict: physical groups
    
    """
    if h2 is None:
        h2 = h1
    if modelname is None:
        modelname = 'mesh_with_crack_edi_'+str(h1)
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(0, 1e-5, 0, h2, 1)
    gmsh.model.geo.addPoint(a, 0, 0, h1, 2) # crack_tip
    gmsh.model.geo.addPoint(width, 0, 0, h2, 3)
    gmsh.model.geo.addPoint(width, height/2, 0, h2, 4)
    gmsh.model.geo.addPoint(0, height/2, 0, h2, 5)
    gmsh.model.geo.addPoint(a-r1, 1e-5, 0, h1, 6)
    gmsh.model.geo.addPoint(a+r1, 0, 0, h1, 7)
    gmsh.model.geo.addPoint(a, r1, 0, h1, 8)
    gmsh.model.geo.addPoint(a-r2, 1e-5, 0, h1, 9)
    gmsh.model.geo.addPoint(a+r2, 0, 0, h1, 10)
    gmsh.model.geo.addPoint(a, r2, 0, h1, 11)

    gmsh.model.geo.addLine(1, 9, 1)
    gmsh.model.geo.addLine(10, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 1, 5)
    gmsh.model.geo.addLine(6, 2, 6)
    gmsh.model.geo.addLine(2, 7, 7)
    gmsh.model.geo.addCircleArc(7, 2, 8, 8)
    gmsh.model.geo.addCircleArc(8, 2, 6, 9)
    gmsh.model.geo.addLine(9, 6, 10)
    gmsh.model.geo.addLine(7, 10, 11)
    gmsh.model.geo.addCircleArc(10, 2, 11, 12)
    gmsh.model.geo.addCircleArc(11, 2, 9, 13)

    gmsh.model.geo.addCurveLoop([1, -13, -12, 2, 3, 4, 5], 1) # Omega \ (contour_area U crack_tip)
    gmsh.model.geo.addCurveLoop([8, 9, 6, 7], 2) # crack_tip
    gmsh.model.geo.addCurveLoop([12, 13, 10, -9, -8, 11], 3) # integration_domain

    gmsh.model.geo.addPlaneSurface([1], 1) # Omega \ (contour_area U crack_tip)
    gmsh.model.geo.addPlaneSurface([2], 2) # crack_tip
    gmsh.model.geo.addPlaneSurface([3], 3) # integration_domain

    dimtags = [(2,3),(2,2),(2,1)]
    gmsh.model.geo.symmetrize(gmsh.model.geo.copy(dimtags), 0., 1., 0., 0.)
    
    gmsh.model.geo.synchronize()
    
    entities_d2 = gmsh.model.getEntities(2)
    entities_d2_cracktip = [(2,2),(2,21)]
    entities_d2_integdom = [(2,3),(2,14)]
    entities_d1_cracktip = [(1,8),(1,9),(1,18),(1,19)]
    entities_d1_integdom = [(1,12),(1,13),(1,15),(1,16)]
    entities_d1_left = [(1,5),(1,33)]
    entities_d1_right = [(1,3),(1,31)]
    entities_d1_bottom = [(1,32)]
    entities_d1_crack = [(1,1),(1, 6),(1,10),(1,27),(1,24),(1,17)]

    gmsh.model.geo.translate(entities_d2, dx=translate[0], dy=translate[1], dz=0.)
    gmsh.model.geo.rotate(entities_d2, x=0, y=0, z=0., ax=0, ay=0, az=1, angle=rotate)
    
    gmsh.model.geo.synchronize()

    physdict = {}
    physdict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [i[1] for i in entities_d2], -1, 
                                                        name='Omega'))
    physdict['integration_domain'] = (2, gmsh.model.addPhysicalGroup(2, 
                                                               [i[1] for i in entities_d2_integdom],
                                                               -1, name='integration_domain'))
    physdict['crack_tip_domain'] = (2, gmsh.model.addPhysicalGroup(2, 
                                                               [i[1] for i in entities_d2_cracktip],
                                                               -1, name='crack_tip_domain'))
    physdict['crack_tip_contour'] = (1, gmsh.model.addPhysicalGroup(1, 
                                                          [i[1] for i in entities_d1_cracktip],
                                                          -1, name='crack_tip_contour'))
    physdict['integration_domain_contour'] = (1, gmsh.model.addPhysicalGroup(1,
                                                          [i[1] for i in entities_d1_integdom],
                                                          -1, name='integration_domain_contour'))
    physdict['crack'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_crack], 
                                                        -1, name='crack'))
    physdict['crack_plane'] = (1, gmsh.model.addPhysicalGroup(1, [2, 11, 7], 
                                                              -1, name='crack_plane'))
    physdict['bottom'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_bottom], -1, 
                                                         name='bottom'))
    physdict['right'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_right], -1, 
                                                        name='right'))
    physdict['top'] = (1, gmsh.model.addPhysicalGroup(1, [4], -1, name='top'))
    physdict['left'] = (1, gmsh.model.addPhysicalGroup(1, [i[1] for i in entities_d1_left], -1, 
                                                       name='left'))
    physdict['crack_tip'] = (0, gmsh.model.addPhysicalGroup(0, [2], -1, name='crack_tip'))
    
    gmsh.model.mesh.generate()
    elemtags = np.concatenate((gmsh.model.mesh.getElements(2, 26)[1][0],
                               gmsh.model.mesh.getElements(2, 14)[1][0],
                               gmsh.model.mesh.getElements(2, 21)[1][0]))
    gmsh.model.mesh.reverseElements(elemtags)

    return modelname, physdict


if __name__ == "__main__":
    mesh_folder = Path("msh")
    mesh_folder.mkdir(exist_ok=True, parents=True)


    ## Generate meshes by splitting for line integration
    W, H, A, R, le = 100., 200., 50., 10., 1.6
    le2 = 10*le

    gmsh.initialize()
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    gmsh.option.setNumber('Mesh.SurfaceEdges', 1 if le > 0.1 else 0)
    mesh_with_crack(W*2, H, A, R, le, le2, translate=(-A,0.), modelname='MT_' + str(le))
    gmsh.write(f'msh/MT_{le}.msh')
    for i in range(1,5):
        gmsh.model.mesh.refine()
        gmsh.write(f'msh/MT_{le/2**i}.msh')
    gmsh.finalize()


    ## Generate meshes by splitting for domain integration
    W, H, A, R1, R2, le = 100., 200., 50., 5., 10., 1.6
    le2 = 10*le

    gmsh.initialize()
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    gmsh.option.setNumber('Mesh.SurfaceEdges', 1 if le > 0.1 else 0)
    mesh_with_crack_edi(W*2, H, A, R1, R2, le, le2, translate=(-A,0.), 
                        modelname='MT_edi_' + str(le))
    gmsh.write(f'msh/MT_edi_{le}.msh')
    for i in range(1,5):
        gmsh.model.mesh.refine()
        gmsh.write(f'msh/MT_edi_{le/2**i}.msh')
    gmsh.finalize()


    ## Generate meshes by splitting for domain integration with R2=0.
    W, H, A, R, le = 100., 200., 50., 10., 1.6
    le2 = 10*le

    gmsh.initialize()
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    gmsh.option.setNumber('Mesh.SurfaceEdges', 1 if le > 0.1 else 0)
    mesh_with_crack_edi_Xtreme(W*2, H, A, R, le, le2, translate=(-A,0.), 
                               modelname='MT_edi_Xtreme10_' + str(le))
    gmsh.write(f'msh/MT_edi_Xtreme10_{le}.msh')
    for i in range(1,5):
        gmsh.model.mesh.refine()
        gmsh.write(f'msh/MT_edi_Xtreme10_{le/2**i}.msh')
    gmsh.finalize()
