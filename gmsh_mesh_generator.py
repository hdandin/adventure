#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:08:33 2023

@author: nchevaug
"""

import gmsh
def square(length=2., relh=1.):
    """ generate a gmsh mesh model of a square of lenght L and mesh density lc """
    modelname = 'ref_square_lc_'+str(relh)
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(-length/2., -length/2., 0, relh, 1)
    gmsh.model.geo.addPoint(length/2., -length/2., 0, relh, 2)
    gmsh.model.geo.addPoint(length/2., length/2., 0, relh, 3)
    gmsh.model.geo.addPoint(-length/2., length/2., 0, relh, 4)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    phys_dict = dict()
    phys_dict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [1], -1, name='Omega'))
    phys_dict['bottom'] = (1, gmsh.model.addPhysicalGroup(1, [1], -1, name='bottom'))
    phys_dict['right'] = (1, gmsh.model.addPhysicalGroup(1, [2], -1, name='right'))
    phys_dict['top'] = (1, gmsh.model.addPhysicalGroup(1, [3], -1, name='top'))
    phys_dict['left'] = (1, gmsh.model.addPhysicalGroup(1, [4], -1, name='left'))
    phys_dict['corner'] = (0, gmsh.model.addPhysicalGroup(0, [1], -1, name='corner'))
    gmsh.model.mesh.generate()
    return modelname, phys_dict

def disk_in_ref_square(length=2., radius=.5, relh=1.):
    """ generate a gmsh mesh model of a square of lenght L containing a centerd disk of radius r
    and mesh density lc """
    modelname = 'disk_in_ref_square'
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(-length/2., -length/2., 0, relh, 1)
    gmsh.model.geo.addPoint(length/2., -length/2., 0, relh, 2)
    gmsh.model.geo.addPoint(length/2., length/2., 0, relh, 3)
    gmsh.model.geo.addPoint(-length/2., length/2., 0, relh, 4)
    lc2 = 1.*relh
    gmsh.model.geo.addPoint(0., 0., 0, lc2, 5)
    gmsh.model.geo.addPoint(0., -radius, 0, lc2, 6)
    gmsh.model.geo.addPoint(radius, 0., 0, lc2, 7)
    gmsh.model.geo.addPoint(0., radius, 0, lc2, 8)
    gmsh.model.geo.addPoint(-radius, 0., 0, lc2, 9)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCircleArc(6, 5, 7, 5)
    gmsh.model.geo.addCircleArc(7, 5, 8, 6)
    gmsh.model.geo.addCircleArc(8, 5, 9, 7)
    gmsh.model.geo.addCircleArc(9, 5, 6, 8)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
    gmsh.model.geo.addPlaneSurface([1, -2], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.synchronize()
    phys_dict = dict()
    phys_dict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [1, 2], -1, name='Omega'))
    phys_dict['bottom'] = (1, gmsh.model.addPhysicalGroup(1, [1], -1, name='bottom'))
    phys_dict['right'] = (1, gmsh.model.addPhysicalGroup(1, [2], -1, name='right'))
    phys_dict['top'] = (1, gmsh.model.addPhysicalGroup(1, [3], -1, name='top'))
    phys_dict['left'] = (1, gmsh.model.addPhysicalGroup(1, [4], -1, name='left'))
    gmsh.model.mesh.generate()
    return modelname, phys_dict

def hole_in_square(length=2., radius=.5, relh=.1):
    """ generate a gmsh mesh model of a square of lenght L containing a
    centered circular hole of radius r. mesh density is lc """
    modelname = 'CircularHoleInRefSquare'
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(-length/2., -length/2., 0, relh, 1)
    gmsh.model.geo.addPoint(length/2., -length/2., 0, relh, 2)
    gmsh.model.geo.addPoint(length/2., length/2., 0, relh, 3)
    gmsh.model.geo.addPoint(-length/2., length/2., 0, relh, 4)
    ratio = (3.14*radius)/length
    lc2 = ratio*relh
    gmsh.model.geo.addPoint(0., 0., 0, lc2, 5)
    gmsh.model.geo.addPoint(0., -radius, 0, lc2, 6)
    gmsh.model.geo.addPoint(radius, 0., 0, lc2, 7)
    gmsh.model.geo.addPoint(0., radius, 0, lc2, 8)
    gmsh.model.geo.addPoint(-radius, 0., 0, lc2, 9)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCircleArc(7, 5, 6, 5)
    gmsh.model.geo.addCircleArc(6, 5, 9, 6)
    gmsh.model.geo.addCircleArc(9, 5, 8, 7)
    gmsh.model.geo.addCircleArc(8, 5, 7, 8)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
    gmsh.model.geo.addPlaneSurface([1, 2], 1)
    gmsh.model.geo.synchronize()
    phys_dict = dict()
    phys_dict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [1], -1, name='Omega'))
    phys_dict['bottom'] = (1, gmsh.model.addPhysicalGroup(1, [1], -1, name='bottom'))
    phys_dict['right'] = (1, gmsh.model.addPhysicalGroup(1, [2], -1, name='right'))
    phys_dict['top'] = (1, gmsh.model.addPhysicalGroup(1, [3], -1, name='top'))
    phys_dict['left'] = (1, gmsh.model.addPhysicalGroup(1, [4], -1, name='left'))
    phys_dict['circle'] = (1, gmsh.model.addPhysicalGroup(1, [5, 6, 7, 8], -1, name='circle'))
    gmsh.model.mesh.generate()
    return modelname, phys_dict

def crack_in_square(relh=1):
    """ generate a gmsh mesh model of a square of lenght 1
    containing an horizontal crack from left to square center """
    modelname = 'square_with_crack'
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(-1, -1, 0, relh, 1)
    gmsh.model.geo.addPoint(1., -1., 0, relh, 2)
    gmsh.model.geo.addPoint(1., 1., 0, relh, 3)
    gmsh.model.geo.addPoint(-1., 1., 0, relh, 4)
    gmsh.model.geo.addPoint(-1., 0., 0, relh, 5)
    gmsh.model.geo.addPoint(0., 1.e-6, 0, relh, 6)
    gmsh.model.geo.addPoint(-1., -1.e-6, 0, relh, 7)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 1, 7)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [1], -1, name='Omega')
    phys_dict = dict()
    phys_dict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [1], -1, name='Omega'))
    phys_dict['dOmega_right'] = (1, gmsh.model.addPhysicalGroup(1, [3, 4], -1, name='dOmega_right'))
    phys_dict['dOmega_left'] = (1, gmsh.model.addPhysicalGroup(1, [7], -1, name='dOmega_left'))
    gmsh.model.mesh.generate()
    return modelname, phys_dict

def l_shape(relh=1.):
    """ generate a gmsh mesh model of a L shaped domain, divided in 3 squares """
    modelname = 'L'
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(0., 0., 0., relh, 1)
    gmsh.model.geo.addPoint(0., -1., 0., relh, 2)
    gmsh.model.geo.addPoint(1., -1., 0., relh, 3)
    gmsh.model.geo.addPoint(1., 0., 0., relh, 4)
    gmsh.model.geo.addPoint(1., 1., 0., relh, 5)
    gmsh.model.geo.addPoint(0., 1., 0., relh, 6)
    gmsh.model.geo.addPoint(-1., 1., 0., relh, 7)
    gmsh.model.geo.addPoint(-1., 0., 0., relh, 8)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 1, 8)
    gmsh.model.geo.addLine(1, 6, 9)
    gmsh.model.geo.addLine(1, 4, 10)
    gmsh.model.geo.addCurveLoop([1, 2, 3, -10], 1)
    gmsh.model.geo.addCurveLoop([10, 4, 5, -9], 2)
    gmsh.model.geo.addCurveLoop([9, 6, 7, 8], 3)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.addPlaneSurface([3], 3)
    gmsh.model.geo.synchronize()
    phys_dict = dict()
    phys_dict['dOmega_left'] = (1, gmsh.model.addPhysicalGroup(1, [7], -1, name='dOmega_left'))
    phys_dict['dOmega_right'] = (1, gmsh.model.addPhysicalGroup(1, [3, 4], -1, name='dOmega_right'))
    phys_dict['dOmega_other'] = (1, gmsh.model.addPhysicalGroup(1, [1, 2, 5, 6, 8], -1,
                                                                name='dOmega_other'))
    phys_dict['dOmega8'] = (1, gmsh.model.addPhysicalGroup(1, [8], -1,
                                                                name='dOmega8'))
    phys_dict['Omega'] = (2, gmsh.model.addPhysicalGroup(2, [1, 2, 3], -1, name='Omega'))
    phys_dict['Omega1'] = (2, gmsh.model.addPhysicalGroup(2, [1], -1, name='Omega1'))
    phys_dict['Omega2'] = (2, gmsh.model.addPhysicalGroup(2, [2], -1, name='Omega2'))
    phys_dict['Omega3'] = (2, gmsh.model.addPhysicalGroup(2, [3], -1, name='Omega3'))
    gmsh.model.mesh.generate()
    #gmsh.model.mesh.setNode(7, np.array([-1,0.,0.]), np.array([], dtype=np.float64))
    return modelname, phys_dict

def ref_tri(relh=1):
    """ generate a gmsh mesh model of the reference triangle """
    modelname = 'ref_tri'
    gmsh.model.add(modelname)
    gmsh.model.geo.addPoint(0., 0., 0., relh, 1)
    gmsh.model.geo.addPoint(1., 0., 0., relh, 2)
    gmsh.model.geo.addPoint(0., 1., 0., relh, 3)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 1, 3)
    gmsh.model.geo.addCurveLoop([1, 2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    phys_dict = dict()
    phys_dict['Omega'] = gmsh.model.addPhysicalGroup(2, [1], -1, name='Omega')
    phys_dict['Omega'] = gmsh.model.addPhysicalGroup(1, [1], -1, name='bottom')
    phys_dict['Omega'] = gmsh.model.addPhysicalGroup(1, [2], -1, name='right')
    phys_dict['Omega'] = gmsh.model.addPhysicalGroup(1, [3], -1, name='left')
    gmsh.model.mesh.generate()
    return modelname, phys_dict
