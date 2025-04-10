#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10th 07:43:08 2024

@author: nchevaug
"""
from __future__ import annotations
from warnings import warn
from collections.abc import Iterable
from typing import Callable, Mapping, Dict
import numpy as np
import numpy.typing as npt
import function_spaces as fs
import oneLevelSimplexMesh as sm

FNDArray = npt.NDArray[np.float64]
INDArray = npt.NDArray[np.int64]

def get_space_entity(edge_groupe_names:Iterable[str],
                     mesh:sm.sMesh)-> Dict[str, INDArray]:
    ''' get the space's edge id corresponding to the edge groupe name'''
    edge_groupe_names2eid: Dict[str, INDArray] = {}
    for name in edge_groupe_names:
        edge_groupe_names2eid[name] = mesh.getEdgeGroupe(name)[0]
    return edge_groupe_names2eid

def get_dofs_and_interpolation_points(
        mesh: sm.sMesh,
        edge_groupe_names: Iterable[str],
        space: fs.FEMSpace
        )->tuple[Dict[str, INDArray], Dict[str, FNDArray]]:
    ''' get the space's degree of freedom id and the corresponding interpolation point
    corresponding to the edge_groupe_names (alist of names). Usefull for Dirichlet BC for example'''
    groupe_name2dofid:Dict[str, INDArray]= {}
    groupe_name2dofxy:Dict[str, FNDArray] = {}
    func_name = get_dofs_and_interpolation_points.__name__
    for phys_name in edge_groupe_names:
        eid, ___ = mesh.getEdgeGroupe(phys_name)
        if eid.size == 0:
            warn('In '+func_name+' no edge found in groupe '+phys_name+'.')
        if space.interpolatoryAt == 'vertex':
            vertices = np.unique((mesh.getEdges2Vertices()[eid]).flatten())
            coord = mesh.getVerticesCoord()[vertices]
            dofsid = space.vertexid2dofid(vertices)
            groupe_name2dofid[phys_name] = dofsid
            groupe_name2dofxy[phys_name] = coord.reshape((-1, 2))
        elif space.interpolatoryAt == 'midEdge':
            groupe_name2dofid[phys_name] = space.edgeid2dofid(eid)
            groupe_name2dofxy[phys_name] = mesh.getEdgesCog()[eid]
        else: raise ValueError(func_name+" not coded for space.interpolatoryAt.")
    return groupe_name2dofid, groupe_name2dofxy

def set_values(vxy: FNDArray,
               evaluator: float|Callable[[FNDArray], FNDArray]
               )->FNDArray:
    """ A simple function that return values associated to the points in array vxy

    according to the evaluator. The evaluator can be a scalar or a function
    """
    func_name = set_values.__name__
    if isinstance(evaluator, float):
        values = np.ones(len(vxy))*evaluator
    elif callable(evaluator):
        values = evaluator(vxy)
    else:
        raise ValueError('evaluator of '+str(type(evaluator))+' unknown in '+func_name+'.')
    return values

dirname2dirindex = {'x':0, 'y':1}
def compute_dirichlet(space:fs.FEMSpace,
                      physname2dirichlet:Mapping[str, Mapping[str, int]],
                      groupename2dofid:Mapping[str, INDArray],
                      groupename2dofxy:Mapping[str, FNDArray]
                      )-> tuple[INDArray, INDArray, FNDArray]:
    ''' return the list of free dofs ids, fixed dofs ids and fixed dofs values '''
    func_name = compute_dirichlet.__name__
    fixed_values = np.array([], dtype=np.float64)
    fixed_dofs = np.array([], dtype=np.int64)
    for phys_name, dir2value in physname2dirichlet.items():
        dofid = groupename2dofid[phys_name]
        dofxy = groupename2dofxy[phys_name]
        for dirname, value in dir2value.items():
            direction = dirname2dirindex[dirname]
            if  direction is None:
                raise ValueError('directionName '+dirname+' unknown in '+func_name+'.')
            fixed_dofs = np.append(fixed_dofs, dofid[:, direction]).astype(np.int64)
            fixed_values = np.append(fixed_values, set_values(dofxy, value)).astype(np.float64)
    #nbfixed = len(fixed_dofs)
    fixed_dofs, index = np.unique(fixed_dofs, return_index=True)
    # if nbfixed != len(fixed_dofs):
    #     warn(' Some dofs where fixed Twice in '+func_name+'.')
    #nbfixed = len(fixed_dofs)
    fixed_values = np.array(fixed_values[index], dtype = np.float64)
    free_dofs = np.setdiff1d(np.arange(space.size()), fixed_dofs).astype(np.int64)
    return free_dofs, fixed_dofs, fixed_values

# def computeNeumann(space, physName2Neumann):
#     ''' return the vector F such as F.T.dot(U) = int_neumann ( t.u_h) dGamma_h '''
#     F = np.zeros((space.scalarSize(), space.vecDim))
#     edgequad = quad.Edge_gauss(0)
#     for physName, dir2Value in physName2Neumann.items():
#         eids, ___ = space.m.getEdgeGroupe(physName)
#         N = space.evalOpOnEdges(eids, edgequad.s)
#         xy = space.m.getEdgesCog(eids)
#         l = space.m.getEdgesLength(eids)
#         t = np.zeros((eids.size, 2))
#         for directionName, value in dir2Value.items():
#             direction = dirname2dirindex[directionName]
#             if  direction is None:
#                 funName = computeNeumann.__name__
#                 raise ValueError('directionName '+directionName+' unknown in '+funName)
#             t[:, direction] = l*set_values(xy, value)
#         F += (edgequad.w*N.T.dot(t.flatten())).reshape((-1, 2))
#     return F.flatten()
