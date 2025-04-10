#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:05:29 2023

@author: nchevaug
"""
import numpy as np

def tri2g(t3, dim = None):
    ''' transform the node coordinate of an array of triangles [..., 3,2|3]
        to an array g[...,2,2|3], where g[...,i, :] is the vector linking vertice 0 to vertice i
        for each each entry in the array ...'''
    assert t3.shape[-2] == 3
    if dim is None:
        dim =t3.shape[-1]
    return (t3[...,1:, :dim] - t3[...,0,np.newaxis,:dim]).reshape(t3.shape[:-2] + (2,dim))

class MappingT3():
    ''' construct mapping from an array of T3 nodes coordinates '''
    def __init__(self, t3xy, hmin=None, alpha=0.1, p=1):
        # pylint: disable=C0103
        self.xy = t3xy
        #self.F = np.swapaxes(self.xy[..., 1:, :]-self.xy[..., [0], :], -1, -2)
        self.F = tri2g(t3xy, 2).swapaxes(1,2)
        F = self.F
        self.cofac_F = np.swapaxes(
            np.array([[F[:, 1, 1], -F[:, 1, 0]], [-F[:, 0, 1], F[:, 0, 0]]]), 0, 2)
        self.J = F[:, 0, 0] * F[:, 1, 1] - F[:, 1, 0] * F[:, 0, 1]
        self.inv_F = self.cofac_F / self.J[:, np.newaxis, np.newaxis]
        if hmin is not None:
            self.hmin = hmin
            self.alpha = alpha
            self.p = 1
            self.Jmin = alpha*((hmin)**p)**2
            limit_reached = self.J < self.Jmin
            self.Jlimited = np.where(limit_reached, self.Jmin, self.J)
            self.inv_F_limited = self.cofac_F/self.Jlimited[:, np.newaxis, np.newaxis]

    def uv2xy(self, uv, elem_ids=None):
        """ map uv local coordinates to xy for all triangles """
        if elem_ids is None:
            return self.xy[..., [0], :] + np.einsum('...jl,...kl->...kj', self.F, uv)
        else:
            return self.xy[elem_ids][..., [0], :] \
                + np.einsum('...jl,...kl->...kj', self.F[elem_ids], uv)

    def xy2uv(self, xy, elem_ids=None):
        """ return uv in each triangle for each entry of xy """
        if elem_ids is not None:
            invFs = self.inv_F[elem_ids] # pylint: disable=C0103
            xy0s = xy - self.xy[elem_ids][..., [0], :]
        else:
            invFs = self.inv_F # pylint: disable=C0103
            xy0s = xy - self.xy[..., [0], :]
        return np.einsum('...jl,...kl->...kj', invFs, xy0s)


class RemapHelper():
    ''' a class to help mapping data when we have a partition of some 'parent element'
        into a set of sons element'''

    def __init__(self, index_parent2sons, sons, parent_mapping, son_mapping):
        self.index_parent2sons = index_parent2sons
        self.sons = sons
        self.parent_mapping = parent_mapping
        self.son_mapping = son_mapping

        self.nbappro = len(index_parent2sons)-1
        self.indextype = index_parent2sons.dtype
        self.appro2nbsons = index_parent2sons[1:] - index_parent2sons[:-1] # pylint: disable=C0103
        # elements for which eval element is the appro element (appro as no sons)
        self.approEQeval = np.where(self.appro2nbsons == 0)[0] # pylint: disable=C0103
        # elements for which eval element is the appro element (appro as no sons)
        self.approNEQeval = np.where(self.appro2nbsons > 0)[0] # pylint: disable=C0103
        self.appro2nbevals = np.where(
            self.appro2nbsons == 0, 1, self.appro2nbsons)
        appro2evals_index = np.zeros(
            len(self.appro2nbevals)+1, dtype=self.indextype)
        np.cumsum(self.appro2nbevals, dtype=self.indextype,
                  out=appro2evals_index[1:])
        self.nbevals = appro2evals_index[-1]
        self.noson2eval_index = appro2evals_index[self.approEQeval]
        self.son2eval_index = np.setdiff1d(
            np.arange(self.nbevals), self.noson2eval_index)
        self.eval2appro = np.repeat(
            np.arange(self.nbappro), self.appro2nbevals)
        self.son2parent_elem_ids = np.repeat(
            self.approNEQeval,  self.appro2nbsons[self.approNEQeval])

    def remap_on_parent(self, evaltrisuv):
        """ compute local coordinate on parent element from local coordinate on son elements """
        sons_point_xy = self.son_mapping.uv2xy(evaltrisuv)[self.sons]
        sons_point_on_parents_uv = self.parent_mapping.xy2uv(
            sons_point_xy, elem_ids=self.son2parent_elem_ids)
        return sons_point_on_parents_uv

    def get_eval_tris(self):
        """ retun an np.array containing the coordinates of the nodes of each sons elements 
            if an element has no sons, it's the coordinates of this element that are returned
            The elements are sorted according to there parent element index
        """
        xytris = np.zeros((self.nbevals,) + (3, 2))
        xytris[self.noson2eval_index] = self.parent_mapping.xy[self.approEQeval]
        xytris[self.son2eval_index] = self.son_mapping.xy[self.sons]
        return xytris


class RemapHelperFiltered():
    """ a class to help mapping data when we have a partition of some 'parent element'
        into a set of sons element.
        This version include a filter on parent element, only on parent
    """

    def __init__(self, index_parent2sons, sons, parent_mapping, son_mapping, only_on_parents):
        self.indextype = index_parent2sons.dtype
        self.nbappro = len(only_on_parents)
        self.only_on_parents = only_on_parents
        self.sons = np.concatenate(np.split(sons, np.column_stack(
            (index_parent2sons[only_on_parents],
             index_parent2sons[only_on_parents+1])).flatten())[1::2])
        self.appro2nbsons = index_parent2sons[only_on_parents +1] \
                            - index_parent2sons[only_on_parents]
        self.parent_mapping = parent_mapping
        self.son_mapping = son_mapping
        self.approEQeval_all = np.where(self.appro2nbsons == 0)[0] # pylint: disable=C0103
        self.approNEQeval_all = np.where(self.appro2nbsons > 0)[0] # pylint: disable=C0103

        # elements for which eval element is the appro element (appro as no sons)
        self.approEQeval = only_on_parents[self.approEQeval_all] # pylint: disable=C0103
        # elements for which eval element is the appro element (appro as no sons)
        self.approNEQeval = only_on_parents[self.approNEQeval_all] # pylint: disable=C0103
        self.appro2nbevals = np.where(
            self.appro2nbsons == 0, 1, self.appro2nbsons)
        appro2evals_index = np.zeros(
            len(self.appro2nbevals)+1, dtype=self.indextype)
        np.cumsum(self.appro2nbevals, dtype=self.indextype,
                  out=appro2evals_index[1:])
        self.nbevals = appro2evals_index[-1]
        self.noson2eval_index = appro2evals_index[self.approEQeval_all]
        self.son2eval_index = np.setdiff1d(
            np.arange(self.nbevals), self.noson2eval_index)
        self.eval2appro = np.repeat(
            np.arange(self.nbappro), self.appro2nbevals)
        self.son2parent_elem_ids = self.only_on_parents[np.repeat(
            self.approNEQeval_all,  self.appro2nbsons[self.approNEQeval_all])]

    def remap_on_parent(self, evaltrisuv):
        """ compute local coordinate on parent element from local coordinate on son elements """
        sons_point_xy = self.son_mapping.uv2xy(evaltrisuv, self.sons)
        sons_point_on_parents_uv = self.parent_mapping.xy2uv(
            sons_point_xy, elem_ids=self.son2parent_elem_ids)
        return sons_point_on_parents_uv

    def get_eval_tris(self):
        """ retun an np.array containing the coordinates of the nodes of each sons elements 
            if an element has no sons, it's the coordinates of this element that are returned
            The elements are sorted according to there parent element index
        """
        xytris = np.zeros((self.nbevals,) + (3, 2))
        xytris[self.noson2eval_index] = self.parent_mapping.xy[self.approEQeval_all]
        xytris[self.son2eval_index] = self.son_mapping.xy[self.sons]
        return xytris



class Mapping3DTri():
    ''' mapping of 3d triangles with 3d coordinates.'''
    def __init__(self, t3):
        assert(t3.shape[-2:] in [(3,2), (3,3)])
        # pylint: disable=C0103
        self.g = tri2g(t3)
        self.m = self.g@self.g.swapaxes(-1,-2)
        self.Q =np.zeros( t3.shape[:-2] + (3,3))
        self.Q[...,:2,:t3.shape[-1]] = self.g
        self.Q[..., 2,:t3.shape[-1]] = np.cross(self.Q[...,0,:], self.Q[...,1,:])
        self.Q[..., 1,:t3.shape[-1]] = np.cross(self.Q[...,2,:], self.Q[...,0,:])
        self.Q /= np.linalg.norm(self.Q, axis = -1, keepdims=True)
        self.F2D = self.Q[..., :-1, :]@self.g.swapaxes(-1,-2)

# if __name__ == '__main__':
#     #pass
# #    tri_xy   = np.array([[0.1,0.3],[1.,0.],[0.5,0.5]])
# #   print('triangle :')
# #   print(tri_xy)
# #   t3m = t3map_old(tri_xy)
# #   print('_____ direct, using map ')
# #   print('area =', np.cross(tri_xy[1] - tri_xy[0], tri_xy[2] - tri_xy[0])/2., t3m.detJ()/2. )
# #   print('F', (tri_xy[1:,:] - tri_xy[0,:]).T, t3m.F())
# #   uv_list = [ np.array([[1./3., 1./3.]]),
# #              np.array([[1./3., 1./3.], [0.,0.], [1.,0.], [0.,1.]])]
# #   for i,  uv in enumerate(uv_list) :
# #       print('test {}'.format(i))
# #       print(' uv')
# #       print(uv)
# #       xy = t3m.uv2xy(uv)
# #       uv = t3m.xy2uv(xy)
# #       print(' xy')
# #       print( xy)
# #       print(' uv')
# #       print( uv)
# #
#     triangles = np.array([[[0.,0., 2.], [1.,0.,2.], [0,1,2]] , [[0.,0.,0.],[1.,0.,0.], [0,1,0]]])
#     #t3 = np.array([[0.1,0.2,0.12], [1.,0.,0.], [0,1,0]])
#     #map = mapping3dtri(tri_xyz)
#     g = tri2g(triangles)
#     mapp = Mapping3DTri(triangles)
#     print(mapp.g)
#     print(mapp.Q)
#     print(mapp.F2D)
