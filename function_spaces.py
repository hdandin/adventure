#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:30:08 2023

@author: nchevaug
"""
#from warnings import warn
from __future__ import annotations
import unittest
from typing import Union
import numpy as np
import numpy.typing as npt
import scipy as sp
import gmsh
import levelSetMeshCut as lscut
import mapping
import oneLevelSimplexMesh as sm
import quadrature as quad
from execTools import Profile
import gmsh_mesh_generator as mg
# import gmshMeshModelInterface as meshInterface
import gmsh_post_pro_interface as gp

FNDArray = npt.NDArray[np.float64]
INDArray = npt.NDArray[np.int64]

def op2sparse(NScalar, nscalar_dofs, a2scalardofs, vecdim, #pylint: disable=C0103
              nshapeperelem, nptperelem, ndataperpoint, only_on=None):
    """ Given a N array (shape function values at integration point for each element
        and  numbering of degree of freedom dat, construct the sparse N Operator (Assembly)
    """
    # nd dataperpoint : not a very good name.
    # my retionnal is this : vecDim represent the number of scalar dof per node.
    # For a scalar field it would be 1, For a 2d vector field, it would be 2,
    # for a 3 d vector Field it would be 2. IT is also possible to have a scalar field, say T
    # and a 3d vector field say, u stored in the same space. the vec dim would then be 4.
    # (The function is meant to assemble a multid field, but with the same space for each dir)
    # The nb data perpoint, is the number of value per vec dim per interpolation point.
    # For the evaluation of the field, it would be 1, for it's gradient dNdu,
    # it would be 2 on aface, 1 on an edge and 3 on a 3d element.
    nelem = len(a2scalardofs)
    neval = len(NScalar)
    npts = nelem*nptperelem
    nrow = npts*ndataperpoint*vecdim
    ncol = nscalar_dofs*vecdim
    shape = (nrow, ncol)
    if only_on is None:
        col = np.repeat(a2scalardofs, nptperelem*ndataperpoint, axis=0)
        row_ptr = np.arange(0, (nrow+1)*nshapeperelem, nshapeperelem)
    else:
        col = np.repeat(a2scalardofs[only_on], nptperelem*ndataperpoint, axis=0)
        row = np.repeat(np.arange(nrow).reshape((nelem, -1))[only_on].flatten(), nshapeperelem)
        row_ptr = np.cumsum(np.bincount(row+1, minlength=nrow+1))
    if vecdim > 1:
        col = col.reshape((-1, nshapeperelem*ndataperpoint))
        col = np.column_stack([vecdim*col + i for i in range(vecdim)]).flatten()
        rshape = (neval, nptperelem, ndataperpoint, nshapeperelem)
        N = np.tile(NScalar.reshape(rshape), [1,1,vecdim,1]).flatten() #pylint: disable=C0103
    else:
        col = col.flatten()
        N = NScalar.flatten() #pylint: disable=C0103
    N = sp.sparse.csr_array((N, col, row_ptr), shape) #pylint: disable=C0103
    N.sort_indices()
    return N

def op2sparse_rmap(NScalar, nscalar_dofs, a2scalardofs, vecdim, #pylint: disable=C0103
                   nshapeperelem, nptperelem,
                   ndataperpoint, rmap, frmap=None, sort=True):
    """ Given a N array (shape function values at integration point for each element 
        (or sub element)) and  numbering of degree of freedom dat, construct 
        the sparse N Operator (Assembly)
        Specialized version for the case (typically xfem) where element are cut in sub cell
    """
    #nelem = len(a2ScalarDofs)
    #neval = len(NScalar)
    #npts = nelem*nptperelem
    #shape = (npts*ndataperpoint*vecDim, nScalarDofs*vecDim)
    npts = rmap.nbevals*nptperelem
    nscalrow = npts*ndataperpoint # nb row for a scalar field
    nrow = nscalrow*vecdim
    ncol = nscalar_dofs*vecdim
    shape = (nrow, ncol)
    if frmap is None:
        col = np.repeat(a2scalardofs, rmap.appro2nbevals*nptperelem*ndataperpoint, axis=0)
        row_ptr = np.arange(0, (nrow+1)*nshapeperelem, nshapeperelem)
        if vecdim > 1:
            col = col.reshape((-1, nshapeperelem*ndataperpoint))
            col = np.column_stack([vecdim*col + i for i in range(vecdim)]).flatten()
            rshape = (rmap.nbevals, nptperelem, ndataperpoint, nshapeperelem)
            N = np.tile(NScalar.reshape(rshape), [1,1,vecdim,1]).flatten() #pylint: disable=C0103
        else:
            col = col.flatten()
            N = NScalar.flatten() #pylint: disable=C0103
    else:
        fscalcol = a2scalardofs[frmap.only_on_parents]
        col = np.repeat(fscalcol, frmap.appro2nbevals*nptperelem*ndataperpoint, axis=0)
        row = np.arange(rmap.nbevals*nptperelem*ndataperpoint).reshape((rmap.nbevals, -1))
        row = row[np.isin(rmap.eval2appro, frmap.only_on_parents)].flatten()
        if vecdim > 1:
            row = np.column_stack(
                tuple(vecdim*row + i for i in range(vecdim))).flatten()
            col = col.reshape((-1, nshapeperelem*ndataperpoint))
            col = np.column_stack([vecdim*col + i for i in range(vecdim)]).flatten()
            rshape = (frmap.nbevals, nptperelem, ndataperpoint, nshapeperelem)
            N = np.tile(NScalar.reshape(rshape), [1,1,vecdim,1]).flatten() #pylint: disable=C0103
            #warn("op2sparse_rmap Not fully tested for case vecDim > 1 and frmap != None")
        else:
            row = row.flatten()
            col = col.flatten()
            N = NScalar.flatten() #pylint: disable=C0103
        row_ptr = np.cumsum(nshapeperelem*np.bincount(row+1, minlength=nrow+1))
    N = sp.sparse.csr_array((N, col, row_ptr), shape) #pylint: disable=C0103
    if sort:
        N.sort_indices()
    return N

def meshcutdata2remap_helper(mesh_cut, hmin=0., alpha=0.1, p=1):
    """ build a RemapHelper Object from mesh_cut data """
    appro_mapping = mapping.MappingT3(
        mesh_cut.xy[mesh_cut.tris], hmin=hmin, alpha=alpha, p=p)
    appro2evals_index, appro2evals = mesh_cut.parents2sons()
    eval_tris_xy = mesh_cut.xy_new_vertices[mesh_cut.new_tris]
    eval_mapping = mapping.MappingT3(eval_tris_xy, hmin=hmin, alpha=alpha, p=p)
    return mapping.RemapHelper(appro2evals_index, appro2evals, appro_mapping, eval_mapping)

def operator_trival2edge(nedge, t2e, t2eflip, ndataperedge=None, datasize=1, dtype=np.float64):
    """ return linear operator that push values at triangles to each edge. useful to compute 
        interelemnt jump for example """
    ntri = len(t2e)
    if ndataperedge is None:
        push2edge_data = np.ones(ntri*datasize*3, dtype=dtype)
        push2edge_row = (2*datasize*t2e + t2eflip).flatten()
        push2edge_col = np.repeat(np.arange(ntri*datasize), 3)
        shape = (2*nedge*datasize, ntri*datasize)
    else:
        push2edge_data = np.ones(ntri*3*ndataperedge*datasize, dtype=dtype)
        push2edge_col = np.arange(ntri*3*ndataperedge*datasize)
        noflip = 2*np.arange(ndataperedge *
                             datasize).reshape((ndataperedge, datasize))
        doflip = noflip[::-1]+1
        push2edge_row = (ndataperedge*datasize*2*t2e[:, :, np.newaxis, np.newaxis] +
                         np.where((t2eflip == 0)[:, :, np.newaxis, np.newaxis],
                                  noflip[np.newaxis, np.newaxis, :, :],
                                  doflip[np.newaxis, np.newaxis, :, :])
                         ).flatten()
        shape = (nedge*ndataperedge*datasize*2, ntri*ndataperedge*3*datasize)
    return sp.sparse.csr_array((push2edge_data, (push2edge_row, push2edge_col)), shape=shape)


def operator_dof2value(space, vecdim, scalar_space_size, tri2dofs, evaltrisuv,
                       rmap=None, frmap=None, raw_data=False, only_on = None):
    """ return a linear operator N that map dof value to value at interpolation point of each 
        element """
    npts = len(evaltrisuv)
    nshape = space.lenN
    Neval = space.N(evaltrisuv) #pylint: disable=C0103
    N = None #pylint: disable=C0103
    if rmap is None and frmap is None:
        if only_on is None:
            N = np.broadcast_to(Neval, (len(tri2dofs), npts, nshape)) #pylint: disable=C0103
        else:
            N = np.broadcast_to(Neval, (len(only_on), npts, nshape)) #pylint: disable=C0103
        if not raw_data:
            N = op2sparse(N, scalar_space_size, tri2dofs, vecdim, #pylint: disable=C0103
                          nshape, npts, 1, only_on=only_on)
    if rmap is not None:
        rmap1 = rmap if frmap is None else frmap
        sons_point_on_parents_uv = rmap1.remap_on_parent(evaltrisuv)
        N = np.zeros((rmap1.nbevals, npts, nshape)) #pylint: disable=C0103
        N[rmap1.noson2eval_index] = Neval
        N[rmap1.son2eval_index] = space.N(sons_point_on_parents_uv)
        if not raw_data:
            N = op2sparse_rmap( #pylint: disable=C0103
                N, scalar_space_size, tri2dofs, vecdim, nshape, npts, 1, rmap, frmap)
    if N is None:
        raise ValueError("N was not constructed due to wrong input")
    return N

def gradNTri(dNdu, uv, invF): #pylint: disable=C0103
    ''' Compute the gradient with regard to space coordinates xy from gradient with regard to
        element coordinate uv'''
    dNdx = np.einsum('...ij,...ik->...jk', invF, dNdu) #pylint: disable=C0103
    # map is constant and DNdU is constant.
    # repeat the value as much as the number of points per elem
    return np.repeat(dNdx[:, np.newaxis, :, :], uv.shape[-2], axis=1)


def operator_dof2grad_tri(space, vecdim, geomdim, scalar_space_size, tri2dofs, trimap, evaltrisuv,
                          rmap=None, frmap=None, only_on=None, raw_data=False, limited=False):
    """ helper function : return operator N that map the dof value to the gradient of field values
    at each point uv for each triangle """
    npts = len(evaltrisuv)
    nshape = space.lenN
    dNdx = None #pylint: disable=C0103
    invF = trimap.inv_F if not limited else trimap.invFLimited #pylint: disable=C0103
    points_uv = np.zeros((0, npts, 0))
    if rmap is None and frmap is None:
        invF = invF if only_on is None else invF[only_on] #pylint: disable=C0103
        dNdx = gradNTri(space.dNdu, points_uv, invF) #pylint: disable=C0103
        if not raw_data:
            dNdx = op2sparse(dNdx, scalar_space_size, tri2dofs, #pylint: disable=C0103
                             vecdim, nshape, npts, geomdim, only_on=only_on)
        elif vecdim!=1:
            neval = len(dNdx)
            nptperelem = npts
            ndataperpoint = geomdim
            nshapeperelem = nshape
            rshape = (neval, nptperelem, ndataperpoint, nshapeperelem)
            dNdx.reshape(rshape)
            rshapev = (neval, nptperelem, ndataperpoint*vecdim, nshapeperelem*vecdim)
            dNdxv = np.zeros(rshapev) #pylint: disable=C0103
            for r in range(vecdim):
                rs = ndataperpoint*r
                re = ndataperpoint*(r+1)
                rows = np.arange(rs,re,1)
                cs = r
                ce = nshapeperelem*vecdim
                cols = np.arange(cs, ce, vecdim)
                dNdxv[..., rows[:, np.newaxis], cols] = dNdx
            dNdx = dNdxv #pylint: disable=C0103
    if rmap is not None:
        # Warning : all coordinate are put to zeros because here we don't care :
        # dNdu is constant on the element
        son_on_parent_uv = np.zeros((0, npts, 0))
        parent_uv = np.zeros((0, npts, 0))
        rmap1 = rmap if frmap is None else frmap
        dNdu = space.dNdu #pylint: disable=C0103
        dNdx = np.zeros((rmap1.nbevals, npts, geomdim, nshape)) #pylint: disable=C0103
        dNdx[rmap1.noson2eval_index] = gradNTri(dNdu, parent_uv, invF[rmap1.approEQeval])
        dNdx[rmap1.son2eval_index] = gradNTri(dNdu, son_on_parent_uv,
                                              invF[rmap1.son2parent_elem_ids])
        if not raw_data:
            dNdx = op2sparse_rmap( #pylint: disable=C0103
                dNdx, scalar_space_size, tri2dofs, vecdim, nshape, npts, geomdim, rmap, frmap)
    if dNdx is None:
        raise ValueError("dNdx was not constructed due to wrong input")
    return dNdx


class FEMSpaceP1():
    """ Classical Linear Finite Element space """
    interpolatoryAt = 'vertex'
    dNdu = np.array([[-1., 1., 0.], [-1., 0., 1.]])
    lenN = 3

    @staticmethod
    def N(uv): #pylint: disable=C0103
        """ Shape function for the triangle """
        u = uv[..., 0]
        v = uv[..., 1]
        return np.stack((1.-u-v, u, v), axis=u.ndim)

    def __init__(self, name:str, mesh:sm.sMesh,
                 vecdim:int=1, tris_map:None|mapping.MappingT3 = None) -> None:
        self.name = name
        self.m = mesh
        self.vecdim = vecdim
        if tris_map is None:
            tris_map = mapping.MappingT3(self.m.getTris2VerticesCoord())
        self.tris_map = tris_map
        self.vid = self.m.getUsedVertices()  # np.unique(tris.flatten())
        if len(self.vid) != len(self.m.getVerticesCoord()):
            self.allnodes = False
            self.vid2dofid = dict(zip(self.vid, np.arange(len(self.vid))))
            self.a2dofs = np.array([[self.vid2dofid[vid] for vid in t]
                                   for t in self.m.getTris2Vertices()])
        else:
            self.allnodes = True
            self.vid2dofid = np.arange(len(self.vid))
            self.a2dofs = self.vid2dofid[self.m.getTris2Vertices()]

    def scalar_size(self):
        """ size (nbdofs) for a scalar space """
        return len(self.vid)

    def size(self):
        """ size (nbdofs) taking into account the vector dim of the field """
        return self.scalar_size()*self.vecdim

    def vertexid2dofid(self, vids):
        """ return the dofs number associated to the vertices id """
        vid_space = np.array([self.vid2dofid[vid] for vid in vids])
        return ((self.vecdim*vid_space)[:, np.newaxis] + np.arange(self.vecdim)).squeeze()

    def edge2edgeid(self, e2v, getxy=False):
        """ from edge defined by there vertices id, recover the edgeid as seen by the space """
        return np.array([], dtype=np.int64).reshape((0, self.vecdim)).squeeze()

    def edgeid2dofid(self, edges):
        """ from edge id recover the associated dofid 
            For FEMSpaceP1, return an empty array
        """
        return np.array([], dtype=np.int64).reshape((0, self.vecdim)).squeeze()

    def interpolate(self, f):
        """ interpolate the function f on to mesh nodes """
        return f(self.m.getVerticesCoord()[self.vid]).flatten()
        # if len(fv.flatten()) != len(self.xy)*self.vecDim  : raise
        # return  fv[self.vid].flatten()

    def operator_dof2val_tri(self, evaltris_uv,
                             rmap=None, frmap=None, only_on=None, raw_data=False):
        """ Return operator N that map the dof value to the field values at each point uv
            for each triangle """
        return operator_dof2value(FEMSpaceP1, self.vecdim, self.scalar_size(), self.a2dofs,
                                  evaltris_uv, rmap=rmap, frmap=frmap,
                                  only_on=only_on, raw_data=raw_data)

    def operator_dof2val_edge(self, eids, evaledge_s=quad.Edge_gauss(0).s):
        """ Return operator N that map the dof value to the field values at each point s of each 
            edge """
        if not self.allnodes:
            raise ValueError("member only coded when all nodes are connected to at \
                least one triangle")
        N = np.tile( #pylint: disable=C0103
            np.stack([1.-evaledge_s, evaledge_s], axis=evaledge_s.ndim).flatten(),
                    eids.size).reshape((eids.size, evaledge_s.shape[0], 2))
        return op2sparse(N, self.scalar_size(), self.m.getEdges2Vertices(eids),
                         self.vecdim, 2, evaledge_s.shape[0], 1)

    def operator_dof2grad_tri(self, evaltrisuv,
                              rmap=None, frmap=None, only_on=None, raw_data=False,
                              limited=False):
        """ Return operator dNdx that map the dof value to the gradient of field values at
            each point uv for each triangle """
        return operator_dof2grad_tri(FEMSpaceP1, self.vecdim, self.m.getGeomDim(),
                                     self.scalar_size(), self.a2dofs, self.tris_map, evaltrisuv,
                                     rmap=rmap, frmap=frmap, only_on=only_on, raw_data=raw_data)

class FEMSpaceP1NC():
    ''' Implementation of cruzeix-raviart Element '''
    interpolatoryAt = 'midEdge'
    dNdu = np.array([[0., 2., -2.], [-2., 2., 0.]])
    lenN = 3

    @staticmethod
    def N(uv): #pylint: disable=C0103
        """ Shape function for the triangle """ 
        u = uv[..., 0]
        v = uv[..., 1]
        return np.stack((1. - 2.*v, 2.*(u + v) - 1, 1 - 2.*u), axis=u.ndim)

    def __init__(self, name, mesh, vecdim=1, tris_map=None) -> None:
        self.name = name
        self.m = mesh
        self.vecdim = vecdim
        self.geomdim = mesh.getGeomDim()
        if tris_map is None:
            tris_map = mapping.MappingT3(mesh.getTris2VerticesCoord())
        self.tris_map = tris_map
        self.a2dofs, __ = mesh.getTris2Edges()

    def scalar_size(self):
        """ size (nbdofs) for a scalar space """
        return self.m.getNbEdges()

    def size(self):
        """ size (nbdofs) taking into account the vector dim of the field """
        return self.scalar_size()*self.vecdim

    def vertexid2dofid(self, vids):
        """ return the dofs number associated to the vertices id """
        return np.array([], dtype=np.int64).reshape((0, self.vecdim)).squeeze()

    def edge2edgeid(self, e2v, getxy=False):
        """ from edge defined by there vertices id, recover the edgeid as seen by the space """
        eid, __ = self.m.findEdgeIds(e2v)
        if not getxy:
            return eid
        return eid, self.m.getEdges2VerticesCoord()[eid]

    def edgeid2dofid(self, eidspace):
        """ from edge id recover the associated dofid """
        return ((self.vecdim*eidspace)[:, np.newaxis] + np.arange(self.vecdim))

    def edge2dofid(self, e2v):
        """ from edge defined by there vertices id, return the associated dofs """
        return self.edgeid2dofid(self.edge2edgeid(e2v))

    def interpolate(self, f)->npt.NDArray[np.float64]:
        """ interpolate the function f on to mesh nodes """
        fe = f(np.sum(self.m.getEdges2VerticesCoord(), axis=1)/2.)
        if len(fe.flatten()) != self.m.getNbEdges()*self.vecdim:
            raise ValueError("size mismatch between f and number of edge dofs")
        return fe.flatten()

    def operator_dof2val_tri(self, evaltrisuv, rmap=None, frmap=None, only_on=None, raw_data=False):
        """ Return operator N that map the dof value to the field values at each point uv
            for each triangle """
        return operator_dof2value(FEMSpaceP1NC, self.vecdim, self.scalar_size(), self.a2dofs,
                            evaltrisuv, rmap=rmap, frmap=frmap, only_on=only_on, raw_data=raw_data)

    def operator_dof2grad_tri(self, evaltrisuv, rmap=None, frmap=None, only_on=None, raw_data=False,
                              limited=False):
        """ Return operator dNdx that map the dof value to the gradient of field values at
            each point uv for each triangle """
        return operator_dof2grad_tri(FEMSpaceP1NC, self.vecdim, self.geomdim, self.scalar_size(),
                                     self.a2dofs, self.tris_map, evaltrisuv,
                                     rmap=rmap, frmap=frmap, only_on=only_on, raw_data=raw_data)

    def operator_trival2edge(self, ndata_per_edge=None, data_size=1):
        """ return linear operator that push values at triangles to each edge. useful to compute 
        interelemnt jump for example """
        return operator_trival2edge(self.m.getNbEdges(), *self.m.getTris2Edges(),
                                    ndataperedge=ndata_per_edge, datasize=data_size)

    def operator_edgetrace(self, quad_edges):
        """ return a linear sparse operator N that compute the trace on each edge at points 
            quad_edges from the dof values.
            t = N@U is an np.array for size nedge*npt*vecdim*2 (it that order)
            the entries i,j k  of t.reshape((nedge, npt, vecdim, 2))
            contain the 2 value at edge i, point j, direction k, which correspond to 
            the values at left and right of the edge (we assume here a 2d manifold)
        """
        npt_per_edge = len(quad_edges.s)
        push2edge = self.operator_trival2edge(ndata_per_edge=npt_per_edge, data_size=self.vecdim)
        evaltrisuv = quad.T3_edges(quad_edges).uv
        N = self.operator_dof2val_tri(evaltrisuv) #pylint: disable=C0103
        return push2edge.dot(N)

    def operator_dof2val_edge(self, eids, evaledges=quad.Edge_gauss(0).s):
        """ Return operator N that map the dof value to the field values at each point s of each 
            edge """
        if evaledges != np.array([.5]):
            raise ValueError("function only coded to eval at mid edge")
        N = np.ones((eids.size, 1, 1)) #pylint: disable=C0103
        return op2sparse(N, self.scalar_size(), eids.reshape((-1, 1)), self.vecdim, 1,
                         evaledges.shape[0], 1)

FEMSpace = Union[FEMSpaceP1, FEMSpaceP1NC]

class Heaviside():
    """ Heaviside shape function defined on a cut mesh """
    def __init__(self, name, mesh_cut_data, ls_tris, ls_tris_cut) -> None:
        self.name = name
        self.side_appro = (1.-lscut.compute_face_side(ls_tris))*0.5
        self.side_eval = (1.-lscut.compute_face_side(ls_tris_cut))*0.5
        if isinstance(mesh_cut_data,mapping.RemapHelper):
            self.rmap = mesh_cut_data
        elif isinstance(mesh_cut_data, sm.meshModification):
            self.rmap = meshcutdata2remap_helper(mesh_cut_data)
        else:
            raise ValueError("Can't interprete mesh_cut_data")

    def eval(self, evaltrisuv, only_on_parents=None, frmap=None):
        """ evaluate the function at each point uv of each element or sub element"""
        if (frmap is not None) and (only_on_parents is not None):
            raise ValueError("can't have both frmap and only_parents be None")
        if only_on_parents is not None:
            frmap = mapping.RemapHelperFiltered(
                self.rmap.index_parent2sons, self.rmap.sons, self.rmap.parent_mapping,
                self.rmap.son_mapping, only_on_parents)
        npts = len(evaltrisuv)
        rmap = self.rmap
        if frmap is None:
            H = np.zeros((rmap.nbevals, npts))  #pylint: disable=C0103
            H[rmap.noson2eval_index] = np.repeat(
                self.side_appro[rmap.approEQeval, np.newaxis], npts, axis=1)
            H[rmap.son2eval_index] = np.repeat(
                self.side_eval[rmap.sons, np.newaxis], npts, axis=1)
        else:  # case where we filter the parent element with frmap.only_on_parent
            Htmp = np.zeros((frmap.nbevals, npts)) #pylint: disable=C0103
            Htmp[frmap.noson2eval_index] = np.repeat(
                self.side_appro[frmap.approEQeval, np.newaxis], npts, axis=1)
            Htmp[frmap.son2eval_index] = np.repeat(
                self.side_eval[frmap.sons, np.newaxis], npts, axis=1)
            H = np.zeros((rmap.nbevals, npts)) #pylint: disable=C0103
            i = np.arange(rmap.nbevals)[np.isin(
                rmap.eval2appro, frmap.only_on_parents)]
            H[i] = Htmp
        return H

class Ridge():
    """ Ridge shape function defined on a cut mesh (usefull for discontinuous gradient) """
    def __init__(self, name, mesh_cut_data, ls_tris, ls_tris_cut) -> None:
        self.name = name
        self.side_appro = (1.-lscut.compute_face_side(ls_tris))*0.5
        self.side_eval = (1.-lscut.compute_face_side(ls_tris_cut))*0.5
        self.abs_phi_appro_vertices = np.abs(ls_tris)
        self.abs_phi_eval_vertices = np.abs(ls_tris_cut)
        if isinstance(mesh_cut_data, mapping.RemapHelper):
            self.rmap = mesh_cut_data
        elif isinstance(mesh_cut_data, sm.meshModification):
            self.rmap = meshcutdata2remap_helper(mesh_cut_data)
        else:
            raise ValueError("Can't interprete mesh_cut_data")

    def eval(self, evaltrisuv, only_on_parents=None, frmap=None, raw_data=False):
        """ evaluate the function at each point uv of each element or sub element"""
        #pylint: disable=C0103
        if only_on_parents is not None:
            frmap = mapping.RemapHelperFiltered(
                self.rmap.index_parent2sons, self.rmap.sons, self.rmap.parent_mapping,
                self.rmap.son_mapping, only_on_parents)
        npts = len(evaltrisuv)
        rmap = self.rmap
        N = FEMSpaceP1.N
        if frmap is None:
            R = np.zeros((rmap.nbevals, npts))
            sons_point_on_parents_uv = rmap.remap_on_parent(evaltrisuv)
            Nabsphi_appro = np.einsum('lik, lk -> li', N(sons_point_on_parents_uv),
                                        self.abs_phi_appro_vertices[rmap.son2parent_elem_ids])
            Nabsphi_eval = np.einsum('...ik, ...jk -> ...ji', N(evaltrisuv),
                                        self.abs_phi_eval_vertices[rmap.sons])
            R[rmap.son2eval_index] = Nabsphi_appro - Nabsphi_eval

        else:
            Rtmp = np.zeros((frmap.nbevals, npts))
            sons_point_on_parents_uv = frmap.remap_on_parent(evaltrisuv)
            Nabsphi_appro = np.einsum('lik, lk -> li', N(sons_point_on_parents_uv),
                                       self.abs_phi_appro_vertices[frmap.son2parent_elem_ids])
            Nabsphi_eval = np.einsum('...ik, ...jk -> ...ji', N(evaltrisuv),
                                      self.abs_phi_eval_vertices[frmap.sons])
            Rtmp[frmap.son2eval_index] = Nabsphi_appro - Nabsphi_eval
            if raw_data:
                return Rtmp
            R = np.zeros((rmap.nbevals, npts))
            index = np.arange(rmap.nbevals)[np.isin(
                rmap.eval2appro, frmap.only_on_parents)]
            R[index] = Rtmp
        return R

    def evalGrad(self, evaltrisuv, only_on_parents=None, frmap=None):
        """ evaluate the gradiant of thefunction at each point uv of each element or sub element"""
        #pylint: disable=C0103
        if only_on_parents is not None:
            frmap = mapping.RemapHelperFiltered(
                self.rmap.index_parent2sons, self.rmap.sons, self.rmap.parent_mapping,
                self.rmap.son_mapping, only_on_parents)
        npts = len(evaltrisuv)
        rmap = self.rmap
        inv_F_parent = rmap.parent_mapping.inv_F
        inv_F_son = rmap.son_mapping.inv_F
        dNdu = FEMSpaceP1.dNdu
        dim = 2
        if frmap is None:
            dRdx = np.zeros((rmap.nbevals, npts, dim))
            dNdxparent = gradNTri(dNdu, evaltrisuv, inv_F_parent[rmap.son2parent_elem_ids])
            dNdxson = gradNTri(dNdu, evaltrisuv, inv_F_son[rmap.sons])
            dNdx_absphi_appro = np.einsum('lijk, lk -> lij', dNdxparent,
                                          self.abs_phi_appro_vertices[rmap.son2parent_elem_ids])
            dNdx_absphi_eval = np.einsum('lijk, lk -> lij', dNdxson,
                                          self.abs_phi_eval_vertices[rmap.sons])
            dRdx[rmap.son2eval_index] = dNdx_absphi_appro - dNdx_absphi_eval
        else:
            dRdxtmp = np.zeros((frmap.nbevals, npts, dim))
            dNdxparent = gradNTri(dNdu, evaltrisuv, inv_F_parent[frmap.son2parent_elem_ids])
            dNdxson = gradNTri(dNdu, evaltrisuv, inv_F_son[frmap.sons])
            dNdx_absphi_appro = np.einsum('lijk, lk -> lij', dNdxparent,
                                          self.abs_phi_appro_vertices[frmap.son2parent_elem_ids])
            dNdx_absphi_eval = np.einsum('lijk, lk -> lij', dNdxson,
                                         self.abs_phi_eval_vertices[frmap.sons])
            dRdxtmp[frmap.son2eval_index] = dNdx_absphi_appro - dNdx_absphi_eval
            dRdx = np.zeros((rmap.nbevals, npts, dim))
            i = np.arange(rmap.nbevals)[np.isin(
                rmap.eval2appro, frmap.only_on_parents)]
            dRdx[i] = dRdxtmp
        return dRdx


class XFEMSpace():
    """ XFEM Function space : input an enrichment function and a base space
        and a level set on which the enrichment is defined
    """
    def __init__(self, name, basespace, enrichment, mesh_cut, ls, ls_cut, rmap=None) -> None:
        self.name = name
        self.rmap = meshcutdata2remap_helper(mesh_cut) if rmap is None else rmap
        self.enrich = enrichment(
            name+'_enrich', self.rmap, ls[mesh_cut.tris], ls_cut[mesh_cut.new_tris])
        self.base = basespace(name+'_base', sm.sMesh(mesh_cut.xy,
                              mesh_cut.tris), tris_map=self.rmap.parent_mapping)
        cut_elem = self.rmap.approNEQeval
        if isinstance(self.enrich, Heaviside):
            venr1 = mesh_cut.tris[cut_elem].flatten()
            venr0 = np.array(np.where(ls == 0.)[0], dtype=venr1.dtype)
            self.enriched_vertices = np.unique(np.hstack((venr0, venr1)))
            self.enriched_support = np.where(
                np.any(np.isin(mesh_cut.tris, self.enriched_vertices), axis=1))[0]
        if isinstance(self.enrich, Ridge):
            self.enriched_support = cut_elem
            self.enriched_vertices = np.unique(mesh_cut.tris[cut_elem])
        self.frmap = mapping.RemapHelperFiltered(self.rmap.index_parent2sons, self.rmap.sons,
                        self.rmap.parent_mapping, self.rmap.son_mapping, self.enriched_support)

    def scalar_size(self):
        """ size (nbdofs) for a scalar space """
        return len(self.enriched_vertices)

    def operator_dof2val_tri(self, evaltrisuv, rmap = None):
        """ Return operator N that map the dof value to the field values at each point uv
            for each triangle """
        # if (rmap != self.rmap):
        #     raise ValueError("Trying to eval XRidgeSpaceP1 with an rmap not \
        #                      conforming to the constructor's rmap")
        base_val = self.base.operator_dof2val_tri(evaltrisuv, self.rmap, self.frmap)
        enr_val = (self.enrich.eval(evaltrisuv, frmap=self.frmap).flatten()).reshape((-1, 1))
        return (enr_val*base_val[:, self.enriched_vertices]).tocsr()

    def operator_dof2grad_tri(self, evaltrisuv, rmap = None):
        """ Return operator dNdx that map the dof value to the gradient of field values at
            each point uv for each triangle """
        # if (rmap != self.rmap):
        #     raise ValueError("Trying to eval XFEMSpace with an rmap not \
        #                      conforming to the constructor's rmap")
        dim = 2
        base_grad = self.base.operator_dof2grad_tri(evaltrisuv, self.rmap, self.frmap)
        enr_val = np.repeat(self.enrich.eval(
            evaltrisuv, frmap=self.frmap).flatten(), dim).reshape((-1, 1))
        dNdx = (enr_val*base_grad[:, self.enriched_vertices]).tocsr() # pylint: disable=C0103
        # grad of heaviside is zero every where....
        if not isinstance(self.enrich, Heaviside):
            grad_enr = self.enrich.evalGrad(evaltrisuv).flatten()
            grad_enr = sp.sparse.csr_array((grad_enr, (np.arange(len(grad_enr)), np.repeat(
                np.arange(len(grad_enr)//2), 2))), shape=(len(grad_enr), len(grad_enr)//2))
            base_val = self.base.operator_dof2val_tri(evaltrisuv, self.rmap, self.frmap)[
                :, self.enriched_vertices]
            dNdx = dNdx + grad_enr.dot(base_val) # pylint: disable=C0103
        return dNdx

class XRidgeSpaceP1():
    """ A special enrichment space for Ridge enrichment """
    def __init__(self, name, mesh_cut, ls, ls_cut, rmap=None, dosvd=False) -> None:
        self.dosvd = dosvd
        self.mesh_cut = mesh_cut
        self.name = name
        self.rmap = meshcutdata2remap_helper(mesh_cut) if rmap is None else rmap
        cut_elem = self.rmap.approNEQeval
        self.enrich = Ridge(name+'_enrich', self.rmap,
                            ls[mesh_cut.tris], ls_cut[mesh_cut.new_tris])
        self.base = FEMSpaceP1(name+'_base', sm.sMesh(mesh_cut.xy,
                               mesh_cut.tris), vecdim=1, tris_map=self.rmap.parent_mapping)
        self.frmap = mapping.RemapHelperFiltered(self.rmap.index_parent2sons, self.rmap.sons,
                                    self.rmap.parent_mapping, self.rmap.son_mapping, cut_elem)

    def scalar_size(self):
        """ size (nbdofs) for a scalar space """
        return len(self.mesh_cut.cut_edge_vertices)

    def operator_dof2val_tri(self, evaltrisuv, rmap=None, raw=False):
        """ Return operator N that map the dof value to the field values at each point uv
            for each triangle """
        # if (rmap != self.rmap):
        #     raise ValueError("Trying to eval XRidgeSpaceP1 with an rmap not \
        #                      conforming to the constructor's rmap")
        nodes_uv = quad.T3_nodes().uv
        enrval_atnodes:np.ndarray = self.enrich.eval(nodes_uv, frmap=self.frmap,
                      raw_data=True)  # .reshape((-1,1))
        nodes_uv = quad.T3_nodes().uv
        npts = len(nodes_uv)
        cut_vertices = self.mesh_cut.cut_vertices
        cut_edges = self.mesh_cut.cut_edge_vertices
        cut_tris1 = self.mesh_cut.new_tris
        edge_funtmp = np.zeros((self.frmap.nbevals, npts, len(cut_edges)))
        for i, v in enumerate(cut_vertices):
            edge_funtmp[self.frmap.son2eval_index, :, i] = np.where(cut_tris1 == v, 1., 0.)[
                self.frmap.sons]
        Nat_nodes_nzelm = enrval_atnodes[:, :, np.newaxis]*edge_funtmp  # pylint: disable=C0103
        nenr = Nat_nodes_nzelm.shape[-1]
        if self.dosvd:
            N = Nat_nodes_nzelm.reshape((-1, len(cut_edges))) # pylint: disable=C0103
            svd = np.linalg.svd(N)
            print('shape : ', N.shape, 'singularvarlues ridge N:', svd[1])
        if np.all(nodes_uv != evaltrisuv):
            Nat_pt_nzelm = np.einsum('ps, ise -> ipe', # pylint: disable=C0103
                                     FEMSpaceP1.N(evaltrisuv), Nat_nodes_nzelm)
        else:
            Nat_pt_nzelm = Nat_nodes_nzelm # pylint: disable=C0103
        if raw is True:
            N = Nat_pt_nzelm # pylint: disable=C0103
        else:
            eval_elements = np.arange(self.rmap.nbevals)[np.isin(
                self.rmap.eval2appro, self.frmap.only_on_parents)]
            npts = len(evaltrisuv)
            row = np.repeat(np.repeat(npts*eval_elements, npts) +
                            np.tile(np.arange(npts), len(eval_elements)), nenr)
            col = np.tile(np.arange(nenr), npts*len(eval_elements))
            N = sp.sparse.csr_array( # pylint: disable=C0103
                (Nat_pt_nzelm.flatten(), (row, col)), (self.rmap.nbevals*npts, nenr))
        return N

    def operator_dof2grad_tri(self, evaltrisuv, rmap = None):
        """ Return operator dNdx that map the dof value to the gradient of field values at
            each point uv for each triangle """
        # if (rmap != self.rmap):
        #     raise ValueError("Trying to eval XRidgeSpaceP1 with an rmap not \
        #                      conforming to the constructor's rmap")
        dim = 2
        npts = len(evaltrisuv)
        nodes_uv = quad.T3_nodes().uv
        nshape = 3
        NRidge = self.operator_dof2val_tri(nodes_uv, raw=True) # pylint: disable=C0103
        nenr = NRidge.shape[-1]
        NRidge = NRidge.reshape((-1, 3, nenr)) # pylint: disable=C0103
        dNdx = np.zeros((self.frmap.nbevals, npts, dim, nshape)) # pylint: disable=C0103
        dNdx[self.frmap.son2eval_index] = gradNTri(
            FEMSpaceP1.dNdu, evaltrisuv, self.frmap.son_mapping.inv_F[self.rmap.sons])
        dNRidgedx = np.einsum('ijkl, ilm ->ijkm', dNdx, NRidge) # pylint: disable=C0103
        eval_elements = np.arange(self.rmap.nbevals)[np.isin(
            self.rmap.eval2appro, self.frmap.only_on_parents)]
        npts = len(evaltrisuv)
        row = np.repeat(np.repeat(npts*dim*eval_elements, npts*dim) +
                        np.tile(np.arange(npts*dim), len(eval_elements)), nenr)
        col = np.tile(np.arange(nenr), dim*npts*len(eval_elements))
        dN = sp.sparse.csr_array( # pylint: disable=C0103
            (dNRidgedx.flatten(), (row, col)), (self.rmap.nbevals*npts*dim, nenr))
        return dN


class SumSpace():
    """ make a space as a sum of different space """
    def __init__(self, list_of_space) -> None:
        self.list_of_space = list_of_space
    def operator_dof2val_tri(self, evaltrisuv, rmap=None):
        """ Return operator N that map the dof value to the field values at each point uv
            for each triangle """
        N = [space.operator_dof2val_tri(evaltrisuv, rmap=rmap) # pylint: disable=C0103
             for space in self.list_of_space]
        return sp.sparse.hstack(N)
    def operator_dof2grad_tri(self, evaltrisuv, rmap=None):
        """ Return operator dNdx that map the dof value to the gradient of field values at
            each point uv for each triangle """
        dNs = [space.operator_dof2grad_tri(evaltrisuv, rmap=rmap) # pylint: disable=C0103
               for space in self.list_of_space]
        return sp.sparse.hstack(dNs)

class SomeScalarField():
    """ an example of scalar field for test purpose """
    def __init__(self, f=1.) -> None:
        self.f = f
    def fun(self, xy):
        """ eval the function at points xy """
        return xy[:, 1]*np.cos(self.f*np.pi*xy[:, 0])

    def grad(self, xy):
        """ eval the gradient of function at points xy """
        gx = -xy[:, 1]*self.f*np.pi*np.sin(self.f*np.pi*xy[:, 0])
        gy = np.cos(self.f*np.pi*xy[:, 0])
        return np.stack((gx, gy), 1)


class LinearScalarField():
    " A linear field in 2d : u(x,y) = a*x + b*y"
    def __init__(self, a=3, b=4) -> None:
        self.a, self.b = a, b

    def fun(self, xy):
        """ eval the function at points xy """
        return self.a*xy[:, 0]+self.b*xy[:, 1]

    def grad(self, xy):
        """ eval the gradient of function at points xy """
        gx = self.a*np.ones(len(xy))
        gy = self.b*np.ones(len(xy))
        return np.stack((gx, gy), 1)


class QuadraticScalarField():
    " A Quadratic field for example purpose "
    def __init__(self, hess = np.eye(2) ) -> None:
        self.hess = 0.5*(hess + hess.T)

    def fun(self, xy):
        """ eval the function at points xy """
        x = xy[:, 0]
        y = xy[:, 1]
        h = self.hess
        return .5*h[0,0]*x**2 + (h[0,1]+h[1,0])*x*y+ .5*h[1,1]*y*y

    def grad(self, xy):
        """ eval the gradient of function at points xy """
        x = xy[:, 0]
        y = xy[:, 1]
        h = self.hess
        gx = h[0,0]*x + (h[0,1]+h[1,0])*x*y
        gy = (h[0,1]+h[1,0])*x + h[1,1]*y
        return np.stack((gx, gy), 1)


class SomeVectorField():
    """ an example of vector field for test purpose """
    def __init__(self, f) -> None:
        self.f = f

    def fun(self, xy):
        """ eval the function at points xy """
        f = self.f
        x = xy[:, 0]
        y = xy[:,1]
        return np.column_stack((y*np.cos(f*np.pi*x), *np.cos(f*np.pi*y)))

    def grad(self, xy):
        """ eval the gradient of function at points xy """
        f = self.f
        gxx = -xy[:, 1]*f*np.pi*np.sin(f*np.pi*xy[:, 0])
        gxy = np.cos(f*np.pi*xy[:, 0])
        gyx = np.cos(f*np.pi*xy[:, 1])
        gyy = -xy[:, 0]*f*np.pi*np.sin(f*np.pi*xy[:, 1])
        return np.column_stack((gxx, gxy, gyx, gyy)).reshape((-1, 2, 2))


class QuadraticVectorField():
    """ an example of Vector field for test purpose """
    def __init__(self) -> None:
        pass

    def fun(self, xy):
        """ eval the function at points xy """
        return 0.1*np.column_stack((xy[:, 0]**2, xy[:, 1]**2))

    def grad(self, xy):
        """ eval the gradient of function at points xy """
        gxx = 0.1*2*xy[:, 0]
        gxy = np.zeros(len(xy))
        gyx = np.zeros(len(xy))
        gyy = 0.1*2*xy[:, 1]
        return np.column_stack((gxx, gxy, gyx, gyy)).reshape((-1, 2, 2))


class LinearVectorField():
    """ an example of Vector field for test purpose """
    def __init__(self) -> None:
        pass

    def fun(self, xy):
        """ eval the function at points xy """
        return np.column_stack((xy[:, 0], 2.*xy[:, 1]))

    def grad(self, xy):
        """ eval the gradient of function at points xy """
        gxx = np.ones(len(xy))
        gxy = np.zeros(len(xy))
        gyx = np.zeros(len(xy))
        gyy = 2.*np.ones(len(xy))
        return np.column_stack((gxx, gxy, gyx, gyy)).reshape((-1, 2, 2))

class TestFunctionSpace(unittest.TestCase):
    """ unit test class function space"""
    def test_gradient(self):
        """ testing computation of gradient """
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)
        #gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
        model_name, physical_dict = mg.square(relh=2.)
        mesh = sm.gmshModel2sMesh(model_name, physical_dict)
        space = FEMSpaceP1("Disp", mesh, vecdim=2)
        h = np.array([[1, 2],[3,4]])
        def funv(coord):
            u = h[0,0]*coord[:,0] + h[0,1]*coord[:,1]
            v = h[1,0]*coord[:,0] + h[1,1]*coord[:,1]
            return np.column_stack([u,v])
        U = space.interpolate(funv) #pylint: disable=C0103
        B = space.operator_dof2grad_tri(quad.T3_gauss(0).uv) #pylint: disable=C0103
        grad_u = (B@U).reshape((-1,2,2))
        err = np.linalg.norm(grad_u-h[np.newaxis])
        self.assertTrue(err< 1.e-12)
        xy = mesh.getVerticesCoord()
        tris = mesh.getTris2Vertices()
        ls = lscut.simpleLevelSetFunction.halfplane([0.2,0.], 0.)
        cut_mesh, __ = lscut.levelSetIsoCut(xy, tris, ls(xy), returnparent = True)
        rmap = meshcutdata2remap_helper(cut_mesh)
        B = space.operator_dof2grad_tri(quad.T3_gauss(0).uv, rmap) #pylint: disable=C0103
        grad_u = (B@U).reshape((-1,2,2))
        err = np.linalg.norm(grad_u-h[np.newaxis])
        self.assertTrue(err< 1.e-12)
        gmsh.finalize()

def main():
    """ A program that test most of the function_space fonctionalities 
        This will be probably moved to units test.
    """
    #pylint: disable=C0103
    def scalfun(xy):
        return QuadraticScalarField().fun(xy)
    def vectfun(xy):
        return QuadraticVectorField().fun(xy)
    elementSize = 0.5
    Profile.doprofile = False
    popUpGmsh = True
    femSpaces = {"P1": FEMSpaceP1, "P1NC": FEMSpaceP1NC}
    tensorTypes = {"Scalar": (1, scalfun), "Vector": (2, vectfun)}
    #testPush2Edge = True # NR
    testNoCut = True
    testCut = True
    testHeaviside = True
    testRidge = True
    testXFEMSpace = True

    gmsh.initialize()
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    #modelName, physical_dict = gmshmesh.generate_L_in3parts(elementSize)
    modelName, physical_dict = mg.square(relh=elementSize)
    mesh = sm.gmshModel2sMesh(modelName, physical_dict)
    xy = mesh.getVerticesCoord()
    tris = mesh.getTris2Vertices()
    trisxy = mesh.getTris2VerticesCoord()
    edgexy = mesh.getEdges2VerticesCoord()
    evalAtCorner = quad.T3_nodes().uv
    evalAtCog = quad.T3_gauss(0).uv
    phi = lscut.simpleLevelSetFunction.disk([0., 0.], 0.4)
    if testNoCut:
        for tensorLabel, (vecDim, fun) in tensorTypes.items():
            for spaceLabel, spaceType in femSpaces.items():
                space = spaceType(spaceLabel, mesh, vecdim=vecDim)
                T:np.ndarray = space.interpolate(fun) #pylint: disable=C0103
                N = space.operator_dof2val_tri(evalAtCorner) #pylint: disable=C0103
                dNdx = space.operator_dof2grad_tri(evalAtCog) #pylint: disable=C0103
                vn = 'testNoCut_'+tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(trisxy, N@T, P0=False, viewname=vn, VectorType="Displacement")
                vn = 'testNoCut_Gradient'+tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(trisxy, dNdx@T, P0=True, viewname=vn, VectorType="Arrow")
                if isinstance(space,FEMSpaceP1NC):
                    N_edge = space.operator_edgetrace(quad.Edge_nodes())
                    valEdge = (N_edge@T).reshape(-1, 2)
                    gp.listPlotFieldLine(edgexy, valEdge[:, 0], P0=False, viewname='EdgeL')
                    gp.listPlotFieldLine(edgexy, valEdge[:, 1], P0=False, viewname='EdgeR')
    if testCut:
        for tensorLabel, (vecDim, fun) in tensorTypes.items():
            for spaceLabel, spaceType in femSpaces.items():
                space = spaceType(spaceLabel, mesh, vecdim=vecDim)
                appro_mapping = space.tris_map
                T = space.interpolate(fun) #pylint: disable=C0103
                # Test on regular Mesh
                N = space.operator_dof2val_tri(evalAtCorner) #pylint: disable=C0103
                dNdx = space.operator_dof2grad_tri(evalAtCog) #pylint: disable=C0103
                vn = 'testCut_OnMesh_'+tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(trisxy, N@T, P0=False, viewname= vn, VectorType="Displacement")
                vn = 'testCut_OnMesh_Gradient'+tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(trisxy, dNdx@T, P0=True, viewname=vn, VectorType="Arrow")
                # Test on filtered Mesh
                onlyon = np.arange(0, mesh.getNbTris(), 2)
                N = space.operator_dof2val_tri(evalAtCorner, only_on=onlyon) #pylint: disable=C0103
                dNdx = space.operator_dof2grad_tri(evalAtCog, only_on=onlyon) #pylint: disable=C0103
                vn = 'testCut_OnFilteredMesh_'+tensorLabel + 'Field_'+spaceLabel
                gp.listPlotFieldTri(trisxy, N@T, P0=False, viewname=vn, VectorType="Displacement")
                vn = 'testCut_OnFilteredMesh_Gradient' + tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(trisxy, dNdx@T, P0=True, viewname=vn, VectorType="Arrow")
                # Test on cut Mesh
                ls = phi(mesh.getVerticesCoord())
                lscut.fit_to_vertices(ls, tris, absfittol=1.e-12, relfittol=1.e-3)
                mesh_mods, lsmods = lscut.levelSetIsoCut(xy, tris, ls, returnparent=True)
                eval_mapping = mapping.MappingT3(mesh_mods.xy_new_vertices[mesh_mods.new_tris])
                appro2evals_index, appro2evals = mesh_mods.parents2sons()
                rmap = mapping.RemapHelper(appro2evals_index, appro2evals, appro_mapping,
                                           eval_mapping)
                cut_trisxy = rmap.get_eval_tris()
                N = space.operator_dof2val_tri(evalAtCorner, rmap) #pylint: disable=C0103
                dNdx = space.operator_dof2grad_tri(evalAtCog, rmap) #pylint: disable=C0103
                vn = 'testCut_OnCutMesh_'+tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(cut_trisxy, N@T, viewname=vn, VectorType="Displacement")
                vn = 'testCut_OnCutMesh_Gradient'+tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(cut_trisxy, dNdx@T, P0=True, viewname=vn, VectorType="Arrow")
                # Test on Filtered cut Mesh
                frmap = mapping.RemapHelperFiltered(appro2evals_index, appro2evals, appro_mapping,
                                                    eval_mapping, onlyon)
                N = space.operator_dof2val_tri(evalAtCorner, rmap, frmap) #pylint: disable=C0103
                dNdx = space.operator_dof2grad_tri(evalAtCog, rmap, frmap) #pylint: disable=C0103
                vn = 'testCut_OnCutFilteredMesh_'+tensorLabel + 'Field_' + spaceLabel
                gp.listPlotFieldTri(cut_trisxy, N@T, P0=False, viewname=vn,
                                    VectorType="Displacement")
                vn = 'testCut_OnCutFilteredMesh_Gradient' + tensorLabel+'Field_'+spaceLabel
                gp.listPlotFieldTri(cut_trisxy, dNdx@T, P0=True, viewname=vn, VectorType="Arrow")
    if testHeaviside or testRidge or testXFEMSpace:
        ls = phi(mesh.getVerticesCoord())
        lscut.fit_to_vertices(ls, mesh.getTris2Vertices(),
                              absfittol=1.e-12, relfittol=1.e-3)
        mesh_mods, lsmods = lscut.levelSetIsoCut(
            mesh.getVerticesCoord(), mesh.getTris2Vertices(), ls, returnparent=True)
        ls_parenttris =  ls[mesh_mods.tris]
    if testHeaviside:
        Hfun = Heaviside("H", mesh_mods, ls_parenttris, lsmods[mesh_mods.new_tris])
        H = Hfun.eval(evalAtCorner)
        onlyon = np.arange(0, mesh.getNbTris(), 2)
        Hfiltered = Hfun.eval(evalAtCorner, only_on_parents=onlyon)
        etrisxy = Hfun.rmap.get_eval_tris()
        vn = "Heaviside_ValueOnCutMesh"
        gp.listPlotFieldTri(etrisxy, H, P0=False, viewname=vn, VectorType="Displacement")
        vn = "Heaviside_ValueOnCutFilteredMesh"
        gp.listPlotFieldTri(etrisxy, Hfiltered, P0=False, viewname=vn, VectorType="Arrow")
    if testRidge:
        Rfun = Ridge("R", mesh_mods, ls_parenttris, lsmods[mesh_mods.new_tris])
        R = Rfun.eval(evalAtCorner)
        dRdx = Rfun.evalGrad(evalAtCog)
        onlyon = np.arange(0, mesh.getNbTris(), 2)
        Rfiltered = Rfun.eval(evalAtCorner, only_on_parents=onlyon)
        dRdxfiltered = Rfun.evalGrad(evalAtCog, only_on_parents=onlyon)
        etrisxy = Hfun.rmap.get_eval_tris()
        vn = 'Ridge_ValueOnCutMesh'
        gp.listPlotFieldTri(etrisxy, R, P0=False, viewname=vn,)
        gp.listPlotFieldTri(etrisxy, Rfiltered, P0=False, viewname='Ridge_ValueOnCutFilteredMesh')
        gp.listPlotFieldTri(etrisxy, dRdx, P0=True, viewname='Ridge_GradientOnCutMesh')
        vn = 'Ridge_GradientOnCutFilteredMesh'
        gp.listPlotFieldTri(etrisxy, dRdxfiltered, P0=True, viewname=vn)
    if testXFEMSpace:
        for enrichment, name in zip([Heaviside, Ridge], ['Heaviside', 'Ridge']):
            xspace = XFEMSpace('test', FEMSpaceP1, enrichment, mesh_mods, ls, lsmods)
            gp.listPlotFieldPoint(mesh_mods.xy[xspace.enriched_vertices], np.ones(
                len(xspace.enriched_vertices)), viewname=name+'_Enriched_vertices', PointSize=7)
            gp.listPlotFieldTri(mesh_mods.xy[mesh_mods.tris[xspace.enriched_support]],
                                np.ones(len(xspace.enriched_support)),
                                P0=True, viewname=name+'_Enriched_support')
            etrisxy = xspace.rmap.get_eval_tris()
            Nenr =  xspace.operator_dof2val_tri(evalAtCorner) #pylint: disable=C0103
            dNenrdx = xspace.operator_dof2grad_tri(evalAtCog) #pylint: disable=C0103
            T = np.ones(Nenr.shape[1]) #pylint: disable=C0103
            vnb = name+'_enrichment'
            gp.listPlotFieldTri(etrisxy, Nenr@T, P0=False, viewname=vnb+'_Value*1')
            gp.listPlotFieldTri(etrisxy, dNenrdx@T, P0=True, viewname=vnb+'_Gradient*1')
            for i in range(min(5, Nenr.shape[1])):
                T = np.zeros(Nenr.shape[1])
                T[i] = 1.
                vn = name + f'_enrichment_Value_{i:d}'
                gp.listPlotFieldTri(etrisxy, Nenr.dot(T), P0=False, viewname=vn)
    if popUpGmsh:
        gmsh.fltk.run()
    gmsh.finalize()
    Profile.print_stats()

if __name__ == '__main__':
    main()
