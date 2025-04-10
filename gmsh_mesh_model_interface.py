#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:14:29 2024
@author: nchevaug
This module is an interface to gmsh. gmsh must be initialized for the function to work.
"""
from __future__ import annotations
import gmsh
import numpy as np
if __name__ == "__main__":
    import gmsh_mesh_generator as meshbuilder
#import numpy.typing as npt

gmshTypeTag2DimTypeNbv = {15:(0, 'point', 1), 1:(1, 'edge', 2),
                          2:(2, 'tri', 3), 3:(2, 'quad', 4), 4:(3, 'tet', 4)}
""" a dictionnary to translate gmsh element type tag to a triplet (dim, elementtypename, nbvertex)
othergmsh elem type can be or in gmsh documentation or there :
https://docs.juliahub.com/GmshTools/9rYp5/0.4.2/element_types/  """

def set_model(func):
    ''' a decorator for function (func) that query gmsh

    the first argument of these func is supposed to be the modelName
    the warper set gmsh current model to modelName, then call func
    then reset the model name to the previous one.
    '''
    def wrapper_set_model(*args, **kwargs):
        gmsh_model_name = args[0]
        old_gmsh_model_name = gmsh.model.getCurrent()
        gmsh.model.setCurrent(gmsh_model_name)
        r = func(*args, **kwargs)
        gmsh.model.setCurrent(old_gmsh_model_name)
        return r
    return wrapper_set_model

@set_model
def nb_vertices(gmsh_model_name): # pylint: disable=W0613
    ''' Return the number of vertices in gmsh_model_name '''
    return len(gmsh.model.mesh.getNodes()[0])

@set_model
def nb_triangles(gmsh_model_name): # pylint: disable=W0613
    ''' Return the number of Linear Triangle in gmsh_model_name '''
    return len(gmsh.model.mesh.getElementsByType(2)[0])

@set_model
def get_vertices_coord(gmsh_model_name): # pylint: disable=W0613
    ''' Return all the vertices coordinate (xy, 2d problems) in mesh
    gmsh_model_name as an np.array of shape (nvertex, 2) '''
    tag, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.array(xyz)
    xy_ = xyz.reshape((-1, 3))[:, :-1]
    return xy_[np.argsort(tag)]

@set_model
def get_triangles(gmsh_model_name): # pylint: disable=W0613
    ''' Return all the linear triangles connectivities in mesh gmsh_model_name
    as an np.array of int of shape (ntris, 3) '''
    tris2v = np.array(gmsh.model.mesh.getElementsByType(2)[1])
    return tris2v.reshape((-1, 3)) -1

@set_model
def get_vertices_on_phys_tag(gmsh_model_name, dim, phystag): # pylint: disable=W0613
    ''' return the nodes id and coordinates of all the nodes of model gmsh_model_name
    classifyied on physical entity dim, phystag '''
    nodes_tag, coord = gmsh.model.mesh.getNodesForPhysicalGroup(dim, phystag)
    nodes_tag = np.array(nodes_tag)
    coord = np.array(coord)
    return nodes_tag-1, coord.reshape((-1, 3))[:, :-1]

@set_model
def get_toplevel_elements_vertices_on_phys_tag(gmsh_model_name, dim, phystag): # pylint: disable=W0613
    ''' get all the element of gmsh_model_name of dimension dim classifyed on dim phystag
        the result is given as dictionnary, indexed by elementype name
        ('point', 'edge', 'tri', 'quad', 'tet' ... etc)
        for each entry of the dictionnary, an np array of shape (n, nv)
        containing for each of the n elements there nv node.
    '''
    tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phystag)
    etype2nodes = {}
    for tag in tags:
        elem_types, ___, elem_node_tags = gmsh.model.mesh.getElements(dim, tag)
        elem_types = np.array(elem_types)
        elem_node_tags = np.array(elem_node_tags)
        for typetag, elem_nodes in zip(elem_types, elem_node_tags):
            edim, name, nv = gmshTypeTag2DimTypeNbv[typetag]
            if edim == dim:
                etype2nodes[name] = np.row_stack((
                    etype2nodes.get(name, np.zeros((0, nv), dtype=np.uint64)),
                    elem_nodes.reshape((-1, nv)) -1))
    return etype2nodes

@set_model
def get_group_name2dim_phys_tag(gmsh_model_name, groupe_names=None): # pylint: disable=W0613
    """ return a dictionary mapping group Name to dim and phystag """
    dic = {gmsh.model.getPhysicalName(*pg): pg for pg in gmsh.model.getPhysicalGroups()}
    if groupe_names is None:
        return  dic
    return {name: dic[name] for name in groupe_names}

if __name__ == '__main__':
    def main():
        ''' a simple test checking how set the set_model decorator works'''
        gmsh.initialize()
        gmsh_model_name1, phys_dict1 = meshbuilder.square(relh=2.)
        gmsh_model_name2, phys_dict2 = meshbuilder.square(relh=1.)
        for  gmsh_model_name, phys_dict in zip([gmsh_model_name1, gmsh_model_name2],
                                            [phys_dict1, phys_dict2]):
            print("### model: ", gmsh_model_name)
            print(f'nv: {nb_vertices(gmsh_model_name):d},\
                  nf: {nb_triangles(gmsh_model_name):d} ')
            print('vertices: ')
            print(get_vertices_coord(gmsh_model_name))
            print('Triangles to vertices: ')
            print(get_triangles(gmsh_model_name))
            print('vertices On bottom physical edge')
            bottom_vertex, bottom_xy = get_vertices_on_phys_tag(gmsh_model_name,
                                                                *phys_dict['bottom'])
            print(bottom_vertex)
            print(bottom_xy)
            print('edges On bottom physical edge')
            e2v = get_toplevel_elements_vertices_on_phys_tag(gmsh_model_name,
                                                             *phys_dict['bottom'])['edge']
            print(e2v)
            print("Dictionnary: GroupeName To dim phystag ")
            print(get_group_name2dim_phys_tag(gmsh_model_name))
        gmsh.finalize()


if __name__ == '__main__':
    main()
