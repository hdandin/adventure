#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:17:21 2023

@author: nchevaug
"""
import numpy as np
import numpy.typing as npt
import gmsh
import oneLevelSimplexMesh as onelevel
from execTools import Profile
import gmsh_mesh_generator as mg
import gmsh_post_pro_interface as gp
import gmsh_mesh_model_interface as meshInterface


FNDArray = npt.NDArray[np.float64]
INDArray = npt.NDArray[np.int64]
DyMesh = onelevel.dmesh
MeshMod = onelevel.meshModification

''' This module deals with cutting a mesh with the iso zero of a level-set '''


class simpleLevelSetFunction():
    @staticmethod
    def disk(center, radius):
        def phi(xy):
            xy = np.atleast_2d(xy)
            x, y = xy[:, 0], xy[:, 1]
            return np.sqrt((x-center[0])**2+(y-center[1])**2) - radius
        return phi

    @staticmethod
    def halfplane(point, theta):
        def phi(xy):
            xy = np.atleast_2d(xy)
            x, y = xy[:, 0], xy[:, 1]
            return (x-point[0])*np.cos(theta) + (y-point[1])*np.sin(theta)
        return phi

    @staticmethod
    def complement(ls):
        def phi(xy):
            return -ls(xy)
        return phi

    @staticmethod
    def union(lss):
        def phi(xy):
            xy = np.atleast_2d(xy)
            allls = np.stack([ls(xy) for ls in lss], 1)
            return np.min(allls, 1)
        return phi

    @staticmethod
    def intersection(lss):
        def phi(xy):
            xy = np.atleast_2d(xy)
            allls = np.stack([ls(xy) for ls in lss], 1)
            return np.max(allls, 1)
        return phi


def compute_face_side(tris_ls:FNDArray):
    ''' return on which size is a tri looking at vertices values of the ls

    given tris_ls an 2d array such as each line i give the value of the levelset at
    the nodes of an element
    return an array containing the side of each element
    (-1, 0, 1, respectivily inside, crossing o out of the iso zero of the ls)'''
    return np.where(np.all(tris_ls >= 0., 1), 1, np.where(np.all(tris_ls <= 0., 1), -1, 0))


def faces2cut(tris_ls:FNDArray):
    ''' return which face need to be cut

    given tris_ls an 2d array such as each line i give the value of
    the levelset at the nodes of an element 
    return the indexes of the elements that needs to be cut'''
    #np.where(np.all(np.stack((np.any(tris_ls > 0, 1), np.any(tris_ls < 0, 1)), 1), 1))[0]
    f2c = np.asarray( np.all(np.stack((np.any(tris_ls > 0, 1), np.any(tris_ls < 0, 1)), 1), 1)).nonzero()[0]
    return f2c


def fit_to_vertices(vls:FNDArray, trisv:INDArray, absfittol:float, relfittol:float):
    ''' fit value of the level set so that iso-zero close to a vertex is snapped to the said vertex.
        input : 
        - vls is an array containing the values of the levelset at each node,
        - trisv is the element connectity (trisv[i,j] return node j of element i) 
        - absfittol : absolute fit tolerance. ls is fit to 0. where abs(ls) < absfittol
        - relfittol : relative fit tolerance.
            value of a level-set at a node is set to zero if one of it's edges 
            would be cut by iso-zero at parameter s < relfittol (edges are parametrized from 0 to 1)
    '''
    vls[np.abs(vls) < absfittol] = 0.
    ev = trisv[:, [2, 0, 0, 1, 1, 2]].reshape((-1, 2))
    evls = vls[ev]
    s_index = np.where(evls[:, 0]*evls[:, 1] < 0)[0]
    iev0ls = evls[s_index, 0]
    iev1ls = evls[s_index, 1]
    vls[ev[s_index[-iev0ls/(iev1ls-iev0ls) < relfittol], 0]] = 0.


def set_levelset(vertices:FNDArray, lsAtVertices:FNDArray, setls):
    # internal function
    for v, ls in zip(vertices, lsAtVertices):
        setls(v, ls)

def fit_to_vertices_old(vertices:FNDArray, getls, setls, absfittol:float, relfittol:float):
    """ this is an old slow version of fit to vertices wiht too much loop.

    still used for crackIsocut and multiLevelSetIsoCut ...
    it work directly on mesh vertices of the dynamic mesh structure ....
    it will desapear soon"""

    for v in vertices:
        lsv = getls(v)
        if np.abs(lsv) < absfittol:
            setls(v, 0.)
    for v in vertices:
        ls0 = getls(v)
        if ls0 == 0.:
            continue
        for e in v.edges:
            ls1 = getls(e.v1) if e.v0 == v else getls(e.v0)
            if ls1*ls0 <= 0.:
                s = -ls0/(ls1-ls0)
                if s < relfittol:
                    setls(v, 0.)
                    break


def docut(e, getls):
    return getls(e.v0)*getls(e.v1) < 0


def cutpos(e, getls):
    v0, v1 = e.v0, e.v1
    ls0, ls1 = getls(v0), getls(v1)
    s = -ls0/(ls1-ls0)
    xy = v0.xy*(1-s) + v1.xy*s
    return s, xy


def build_interpolatecallback_default(setls):
    def interpolatecallback_default(s, vn, e):
        setls(vn, 0.)
    return interpolatecallback_default


def facecallback_propagate_parent(f, f0, f1):
    parent = f.id if not hasattr(f, "parent") else f.parent
    e = DyMesh.getCommonEdge(f0, f1)
    f0.parent, f1.parent, e.parent = [parent]*3

def lscut(m, getls, setls, fit=True, absfittol=1.e-6, relfittol=1.e-2,
          interpolatecallback=None, edgecallback=DyMesh.edgeCutCallBackPass,
          facecallback=DyMesh.faceCutCallBackPass):
    if interpolatecallback is None:
        interpolatecallback = build_interpolatecallback_default(setls)
    if fit:
        fit_to_vertices_old(m.getVertices(), getls,
                            setls, absfittol, relfittol)
    edges2cut = [(e, cutpos(e, getls))
                 for e in m.getEdges() if docut(e, getls)]
    edge2cutv = np.zeros((len(edges2cut), 2), dtype=np.dtype('int64'))
    cutv = np.zeros((len(edges2cut), 1), dtype=np.dtype('int64'))

    for ie, (e, (s, xy)) in enumerate(edges2cut):
        vn = m.split_edge(e, xy, edgecallback, facecallback)
        interpolatecallback(s, vn, e)
        edge2cutv[ie] = np.array([e.v0.id, e.v1.id])
        cutv[ie] = vn.id
    return edge2cutv, cutv

def levelSetIsoCut(xy:FNDArray, tris2v:INDArray, vls:FNDArray, returnparent=True):
    ''' improved version of levelSetIsoCut : only the part of the mesh that really need to be cut are loaded in a dynamic mesh. Much faster 
        return improved data structure : meshModification
    '''
    modifs = MeshMod(xy, tris2v)
    # tris_ls[i,j] contains the values of the level set at node j of element i
    tris_ls = vls[tris2v]
    tris2cut = faces2cut(tris_ls)              # list of elements to cut
    modifs.deleted_tris = tris2cut
    # trisZcutv[i,j] : vertices j of triangle to cut i
    tris2cutv = tris2v[tris2cut]
    # list of vertices connected to triangles to cut. also used to remap from new to old numbering
    v2cut = np.unique(tris2cutv)
    modifs.duplicated_vertices = v2cut
    if len(v2cut) == 0:
        neworder = np.zeros(0, dtype=np.uint64)
    else:
        neworder = np.zeros(int(np.max(v2cut)+1), dtype=np.uint64)
    neworder[v2cut] = np.arange(len(v2cut))
    def getls(v):
        return v.ls
    def setls(v, ls): v.ls = ls
    m = DyMesh(xy[v2cut], neworder[tris2cutv])
    set_levelset(m.getVertices(), vls[v2cut], setls)
    if returnparent:
        cutedgevertices_neworder, cut_vertices = lscut(
            m, getls, setls, fit=False, facecallback=facecallback_propagate_parent)
    else:
        cutedgevertices_neworder, cut_vertices = lscut(
            m, getls, setls, fit=False)
    xy_cut_, tris2v_cut_ = m.getCoordAndConnectivity()
    modifs.new_tris = tris2v_cut_
    modifs.xy_new_vertices = xy_cut_
    modifs.cut_edge_vertices = v2cut[cutedgevertices_neworder]
    modifs.cut_vertices = cut_vertices
    vls_cut_ = np.array([v.ls for v in m.getVertices()])
    if returnparent:
        parent_id_neworder = np.array([f.id if not hasattr(
            f, "parent") else f.parent for f in m.getFaces()], dtype=np.int64)
        modifs.new_tris_parent = tris2cut[parent_id_neworder]
        return modifs, vls_cut_
    return modifs, vls_cut_


def multiLevelSetIsoCut(xy, tris, listoflsAtVertices, absfittol=1.e-12, relfittol=1.e-6):
    def getls(i):
        def getlsi(v):
            return v.ls[i]
        return getlsi

    def setls(i):
        def setlsi(v, ls):
            v.ls[i] = ls
        return setlsi
    m = DyMesh(xy, tris)
    nls = len(listoflsAtVertices)
    for v in m.getVertices():
        v.ls = np.zeros(nls)
    for f in m.getFaces():
        f.side = np.zeros(nls)
    for i in range(nls):
        set_levelset(m.getVertices(), listoflsAtVertices[i], setls(i))
    for i in range(nls):
        def interpolatecallback(s, vn, e):
            vn.ls = np.zeros(nls)
            vn.ls[i] = 0.
            for k in set(range(nls)) - {i}:
                vn.ls[k] = e.v0.ls[k]*(1-s) + e.v1.ls[k]*s
        lscut(m, getls(i), setls(i), True, absfittol, relfittol,
              interpolatecallback=interpolatecallback)
    xy_cut, tri_cut = m.getCoordAndConnectivity()
    ls_cut = np.array([v.ls for v in m.getVertices()])
    side_cut = np.hstack([compute_face_side(
        ls_cut[:, i][tri_cut]).reshape((-1, 1)) for i in range(nls)])
    return xy_cut, ls_cut, tri_cut, side_cut


def crackIsoCut(xy, tris, lsn, lst, r=0.5, fit2vertices=True, absfittol=1.e-12, relfittol=1.e-6):
    def setLsn(v, lsn):
        v.lsn = lsn

    def getLsn(v):
        return v.lsn

    def setLst(v, lst):
        v.lst = lst

    def getLst(v):
        return v.lst

    m = DyMesh(xy, tris)
    set_levelset(m.getVertices(), lsn, setLsn)
    set_levelset(m.getVertices(), lst, setLst)

    def interpolatecallback_lsncut(s, vn, e):
        vn.lsn = 0.
        vn.lst = e.v0.lst*(1-s) + e.v1.lst*s

    def interpolatecallback_lstcut(s, vn, e):
        vn.lst = 0.
        vn.lsn = e.v0.lsn*(1-s) + e.v1.lsn*s
        parent = e if not hasattr(e, "parent") else e.parent
        vn.parent = parent

    def edgecallback(e, e0, e1):
        parent = e if not hasattr(e, "parent") else e.parent
        vn = DyMesh.getCommonVertex(e0, e1)
        vn.parent, e0.parent, e1.parent = [parent]*3

    def facecallback(f, f0, f1):
        parent = f.id if not hasattr(f, "parent") else f.parent
        e = DyMesh.getCommonEdge(f0, f1)
        f0.parent, f1.parent, e.parent = [parent]*3
    lscut(m, getLsn, setLsn, True, absfittol, relfittol,
          interpolatecallback_lsncut, edgecallback, facecallback)
    lscut(m, getLst, setLst, True, absfittol, relfittol,
          interpolatecallback_lstcut, edgecallback, facecallback)
    xy_cut, tri_cut = m.getCoordAndConnectivity()
    lsn_cut = np.array([v.lsn for v in m.getVertices()])
    lst_cut = np.array([v.lst for v in m.getVertices()])
    siden_cut = compute_face_side(lsn_cut[tri_cut])
    sidet_cut = compute_face_side(lst_cut[tri_cut])
    parent_id = np.array([-1 if not hasattr(f, "parent")
                         else f.parent for f in m.getFaces()])
    return xy_cut, lsn_cut, lst_cut, tri_cut, parent_id, siden_cut, sidet_cut


def test_suite(*, element_size=0.2, profile=False):
    popUpGmsh = True
    Profile.doprofile = profile
    testFitToVertex = True
    testLevelSetIsoCut = True
    testMultiLevelSetIsoCut = True
    testCrackIsoCut = True
    gmsh.initialize()
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    modelName, _ = mg.l_shape(element_size)
    xy_appro = meshInterface.get_vertices_coord(modelName)
    tris_appro = meshInterface.get_triangles(modelName)
    tris_appro_xy = xy_appro[tris_appro]
    print('nb elements =', len(tris_appro))
    print('nb vertices =', len(xy_appro))
    if testFitToVertex:
        # tests with large value of relfittol (relative (toedges size) fit to vertex tolerance)
        absfittol = 1.e-6
        relfittol = 0.3
        # very small test to check implem. result should be ('fit', [1.,2.,0.,0.])
        tris = np.array([[0, 1, 2], [2, 1, 3]])
        vls = np.array([1., 2., 0., 0.])
        fit_to_vertices(vls, tris, absfittol, relfittol)
        print('fit', vls)
        # Next is a test on the loaded mesh. First we plot the level-set before the fit then after?
        ls = simpleLevelSetFunction.disk([0., 0.], .5)
        vls = ls(xy_appro)
        gp.listPlotFieldTri(tris_appro_xy, vls[tris_appro], P0=False, viewname='ls_before_fit',
                            IntervalsType="Filled iso-Values", NbIso=2, Range=[-1., 1.])
        f2v = tris_appro
        Profile.enable()
        fit_to_vertices(vls, f2v, absfittol, relfittol)
        Profile.disable()
        gp.listPlotFieldTri(tris_appro_xy, vls[tris_appro], P0=False, viewname='ls_after_fit',
                            IntervalsType="Filled iso-Values", NbIso=2, Range=[-1., 1.])
        m = DyMesh(xy_appro, tris_appro)
        vls = ls(xy_appro)

        def getls(v):
            return v.ls

        def setls(v, ls):
            v.ls = ls

        set_levelset(m.getVertices(), vls, setls)
        fit_to_vertices_old(m.getVertices(), getls,
                            setls, absfittol, relfittol)
        xy_cut, tris_cut = m.getCoordAndConnectivity()
        # ls_cut = np.array([getls(v) for v in m.getVertices()])
        gp.listPlotFieldTri(tris_appro_xy, vls[tris_appro], P0=False, viewname='ls_after_fit_old',
                            IntervalsType="Filled iso-Values", NbIso=2, Range=[-1., 1.])
    if testLevelSetIsoCut:
        ls = simpleLevelSetFunction.disk([0., 0.], .5)
        vls = ls(xy_appro)
        fit_to_vertices(vls, tris_appro, 1.e-6, 5.e-3)
        Profile.enable()
        # 3470091 noeud (element size = 1.e-3) -> 1.13 second 2 cut.
        modifs, vls_cut = levelSetIsoCut(
            xy_appro, tris_appro, vls, returnparent=True)
        Profile.disable()
        xy, tris, vls = modifs.getNewMesh(vls, vls_cut)
        gp.listPlotFieldTri(xy[tris], vls[tris], P0=False, viewname="isocut_ls",
                            IntervalsType="Filled iso-Values", NbIso=10)
    if testMultiLevelSetIsoCut:
        ls1 = simpleLevelSetFunction.disk([0., 0.], .5)
        ls2 = simpleLevelSetFunction.disk([0., 0.], .7)
        ls3 = simpleLevelSetFunction.halfplane([0.3, 0.], 0.)
        lss_xy = [ls1(xy_appro), ls2(xy_appro), ls3(xy_appro)]
        Profile.enable()
        xy, mphi, tris, mside = multiLevelSetIsoCut(
            xy_appro, tris_appro, lss_xy, absfittol=0., relfittol=0.02)
        Profile.disable()
        for i in range(len(lss_xy)):
            gp.listPlotFieldTri(xy[tris], mphi[:, i][tris],
                                P0=False, viewname=f"multi_isocut_ls_{i}")
            gp.listPlotFieldTri(xy[tris], mside[:, i],
                                P0=True, viewname=f"multi_isocut_side_{i}")
    if testCrackIsoCut:
        Profile.enable()
        lsn = simpleLevelSetFunction.halfplane([0., 0.5], np.pi/2.)
        lst = simpleLevelSetFunction.intersection([simpleLevelSetFunction.halfplane(
            [.5, 0.], 0.), simpleLevelSetFunction.halfplane([-.5, 0.], np.pi)])
        xy_cut, lsn_cut, lst_cut, tri_cut, parent_id, siden_cut, sidet_cut = crackIsoCut(
            xy_appro, tris_appro, lsn(xy_appro), lst(xy_appro), r=0.5,
            absfittol=1.e-12, relfittol=1.e-6)
        Profile.disable()
        elem_id = np.arange(0, len(tris_appro))
        gp.listPlotFieldTri(xy_cut[tri_cut], lsn_cut[tri_cut], P0=False,
                            viewname="crackisocut_lsn", IntervalsType="Filled iso-Values")
        gp.listPlotFieldTri(xy_cut[tri_cut], siden_cut, P0=True,
                            viewname="crackisocut_lsn_side", IntervalsType="Filled iso-Values")
        gp.listPlotFieldTri(xy_cut[tri_cut], lst_cut[tri_cut], P0=False,
                            viewname="crackisocut_lst", IntervalsType="Filled iso-Values")
        gp.listPlotFieldTri(xy_cut[tri_cut], sidet_cut, P0=True,
                            viewname="crackisocut_lst_side", IntervalsType="Filled iso-Values")
        gp.listPlotFieldTri(tris_appro_xy, elem_id, P0=True,
                            viewname="face_id", IntervalsType="Numeric values")
        gp.listPlotFieldTri(xy_cut[tri_cut], parent_id, P0=True,
                            viewname="parentface_id", IntervalsType="Numeric values")
    if popUpGmsh:
        gmsh.fltk.run()
    gmsh.finalize()
    Profile.print_stats()


if __name__ == '__main__':
    test_suite()
