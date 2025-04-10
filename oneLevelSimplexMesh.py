#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:26:58 2023

@author: nchevaug
"""

# using numpy array only to store the data#

import numpy as np
from itertools import count

def _vertices2vertices(e2v, v2e_index, v2e):
    nv = len(v2e_index)-1
    v0 = np.repeat(np.arange(nv), v2e_index[1:]-v2e_index[:-1])
    tmp = e2v[v2e]
    v2v = np.where(tmp[:,0] == v0, tmp[:,1], tmp[:,0])
    return v2e_index, v2v
    
def createEdgesFromTris(t2v):
    ''' from a list of an array of triangles, defined by its node id, return :
         - e2v : an array of ne unique edges bounding each triangle, 
           defined by a np.array of shape (ne, 2). each line j of sube contain vid0 and vid1, the vertices id defining edge j, such as vid0 < vid1.
         - t2e an array of shape (nt, 3), containing for each triangle the id of each of it's 3 unique edges.
           By unique we mean here that edge i->j is the same as edge j->i
         - t2eflip an array of shape (nt, 3), containing for each triangle 3 entries ( 0  or 1), telling if edge i of the local numbering of the triangle id
           "flipped" compared to the global edge id  it refer to in t2e
    '''
    t2ev       = t2v[:, np.array([[0,1], [1,2], [2,0]])]
    e2v        = t2ev.reshape(-1, 2)
    asort      = e2v.argsort(axis=1)
    e2v, emap  = np.unique(np.take_along_axis(e2v, asort, 1), axis=0, return_inverse=True)
    t2e        = emap.reshape((t2v.shape[0],3))
    t2eflip    = np.where(t2ev[:, :, 0] > t2ev[:, :, 1], 1, 0) 
    return e2v, t2e, t2eflip

def upwardAdjacencies(t):
    flatt = t.flatten()
    order = np.argsort(flatt)
    fo = flatt[order]
    fo = np.array(fo, dtype  =np.int64)
    bc = np.bincount(fo)
    elementstart = np.cumsum(np.hstack((np.array([0], dtype = 'int'),bc)))        
    element = order//t.shape[1]
    return  elementstart, element

class sMesh():
    def __init__(self, xy, tris):
        self.xy = xy
        self.t2v  = tris
        self.groupe2vertices = {}
        self.groupe2edges = {}
        self.groupe2tris = {}
    def setVertexGroupe(self, groupeName, v, warnExisting = True, save = True):
        if warnExisting and (groupeName in self.groupe2vertices):
            print('warning : vertexGroupe ', groupeName, 'already exist in sMesh')
        self.groupe2vertices[groupeName] = v
    def setEdgeGroupeFromEdge2VertexId(self, groupeName, e2v, warnExisting = True, save = True):
        if warnExisting and (groupeName in self.groupe2edges):
            print('warning : edgeGroupe ', groupeName, 'already exist in sMesh')
        self.groupe2edges[groupeName] = self.findEdgeIds(e2v) 
    def setTriGroupeFromTri2VertexId(self, groupeName, t2v, warnExisting = True, save = True):
        if warnExisting and (groupeName in self.groupe2tris):
            print('warning : triGroupe ', groupeName, 'already exist in sMesh')
        self.groupe2tris[groupeName] = self.findTriIds(t2v) 
    def getVertexGroupe(self, groupeName):
        return self.groupe2vertices.get(groupeName, np.array([], dtype = np.int64))
    def getEdgeGroupe(self, groupeName):
        return self.groupe2edges.get(groupeName, (np.array([], dtype = np.int64), np.array([], dtype = np.int64)))
    def getTriangleGroupe(self, groupeName):
        return self.groupe2tris.get(groupeName, (np.array([], dtype = np.int64), np.array([], dtype = np.int64)))
    def getEdgeGroupeNames(self):
        return list(self.groupe2edges.keys())
    def getNbTris(self):
        return self.t2v.shape[0]
    def getNbVertices(self, save = True):
        return self.getUsedVertices(save = save).shape[0]
    def getGeomDim(self):
        if self.xy.shape[1] != 2 :
            raise ValueError("Error : code tested only for geomDim == 2 (triangle in a plane ...) ")
        return 2    
    def getUsedVertices(self, *, save = True):
        if hasattr(self, 'vid') : vid = self.vid
        else: 
            vid = np.unique(self.t2v.flatten())
            if save : self.vid = vid
        return vid
    def getTris2Vertices(self) :
        return self.t2v
    def getTris2VerticesCoord(self, *, save = True) :
        if hasattr(self, 't2vxy'): t2vxy = self.t2vxy
        else :
            t2vxy = self.xy[self.t2v]
            if save : self.t2vxy = t2vxy
        return t2vxy
    def getTris2VerticesCog(self, *, save = True):
        return np.sum(self.getTris2VerticesCoord(save=save), axis = 1)/3.
    def getEdges2VerticesCoord(self, eids = None, *, save = True) :
        if hasattr(self,'e2vxy'):  e2vxy = self.e2vxy
        else :
            e2vxy = self.xy[self.getEdges2Vertices()]
            if save : self.e2vxy = e2vxy
        if eids is None  : return e2vxy
        return e2vxy[eids]
    def getEdgesCog(self, eids = None):
        e2xy = self.getEdges2VerticesCoord(eids)
        cog  =  0.5*np.sum(e2xy, axis = 1)
        return cog
    def getEdgesNormal(self, eids = None, flips = None):
        if (self.getGeomDim() != 2) : raise 
        e2xy = self.getEdges2VerticesCoord(eids)
        e2dir      = e2xy[:,1,:] - e2xy[:,0,:]
        e2dirnorm  = np.linalg.norm(e2dir, axis =1) 
        normal = np.column_stack((e2dir[:,1]/e2dirnorm, -e2dir[:,0]/e2dirnorm))
        if flips is None : return normal
        else  :            return np.where( (flips == 0)[:, np.newaxis], normal, -1.*normal)
    def getEdgesLength(self, eids = None, *, save = True):
        if hasattr(self, 'elength') : elength = self.elength
        else :
             e2vxy   = self.getEdges2VerticesCoord(eids = None, save = save)
             elength = np.linalg.norm( e2vxy[:, 1,:] - e2vxy[:, 0, :] , axis = 1)
             if save : self.elength = elength
        if eids is None  :return elength
        return elength[eids]
    def getVerticesCoord(self):
        return self.xy
    def _e2v_t2e(self, save = True):
        if hasattr(self, 'e2v') : e2v, t2e, t2eflip = self.e2v, self.t2e, self.t2eflip 
        if not hasattr(self, 'e2v') :
             e2v, t2e, t2eflip = createEdgesFromTris(self.t2v)
             if save : self.e2v, self.t2e, self.t2eflip = (e2v, t2e, t2eflip)
        return e2v, t2e, t2eflip
    def getEdges2Vertices(self, eids = None, *, save = True):
        e2v, __, __ = self._e2v_t2e(save = save)
        if eids is None : return e2v
        return e2v[eids]
    def getTris2Edges(self, flip = True, *, save = True):
        __, t2e, t2eflip = self._e2v_t2e(save = save)
        if flip :  return t2e, t2eflip
        else : return t2e
    def getNbEdges(self, *, save = True):
        e2v, __, __ = self._e2v_t2e(save = save)
        return e2v.shape[0]
    def _e2t_(self, save = True):
        if hasattr(self, 'e2t_r'): e2t_r, e2t = (self.e2t_r, self.e2t)
        else:
            __, t2e, __ = self._e2v_t2e(save = save)
            e2t_r, e2t = upwardAdjacencies(t2e) 
            if save : self.e2t_r, self.e2t = (e2t_r, e2t)
        return e2t_r, e2t
    def getBoundaryEdges(self, *, save = True):
        if hasattr(self, 'beid'): beid = self.beid
        else:
            e2t_r, e2t = self._e2t_(self, save)
            beid = np.where( (e2t_r[1:] - e2t_r[:-1]) == 1)[0]
            if save : self.beid = beid
        return beid
    def getInternalEdges(self, *, save = True):
        if hasattr(self, 'ieid'): ieid = self.ieid
        else:  
            e2t_r, e2t = self._e2t_(save)
            ieid = np.where( (e2t_r[1:] - e2t_r[:-1]) == 2)[0]
            if np.any ((e2t_r[1:] - e2t_r[:-1]) > 2) : raise
            if save : self.ieid = ieid
        return ieid
    def findEdgeIds(self, e2v):
        e2v_sorted            = np.sort(e2v, axis=1)
        e2v_mesh              = self.getEdges2Vertices()
        indexes_v0_start_inmesh      =  np.searchsorted(e2v_mesh[:,0], e2v_sorted[:,0], side='left')
        indexes_v0_end_inmesh        =  np.searchsorted(e2v_mesh[:,0], e2v_sorted[:,0], side='right')
        eids  = []
        eflips = []
        for i, v1 in enumerate(e2v_sorted[:,1]) :
            s = indexes_v0_start_inmesh[i]
            e = indexes_v0_end_inmesh[i]
            eid = s+np.searchsorted(e2v_mesh[s:e,1], [v1], side='left')
            if s<= eid < e :
                eids.append(eid[0])
                eflips.append(not(np.all( e2v_mesh[eid] == e2v[i])))
        return np.array(eids), np.array(eflips, dtype =np.int64)
    def findTriIds(self, t2v):
        t2v_gmsh              = np.sort(t2v, axis=1)
        t2v_mesh              = np.sort(self.getTris2Vertices(), axis=1)
        tids = []
        if np.all(len(t2v_gmsh) == len(t2v_mesh)):
            tids = [i for i in range(len(t2v_mesh))]
        else:
            for i,t in enumerate(t2v_gmsh):
                tid = np.where((t == t2v_mesh).all(axis=1))[0]
                tids.append(tid[0])
        return np.array(tids)

def gmshModel2sMesh(modelName, groupeNames = []):    
    import gmsh_mesh_model_interface as mi
    xy             = mi.get_vertices_coord(modelName)
    tris           = mi.get_triangles(modelName)
    mesh           = sMesh(xy, tris)
    groupe2PhysTag = mi.get_group_name2dim_phys_tag(modelName,  groupeNames)
    for name, (dim, physTag) in groupe2PhysTag.items() :
        if dim == 0:
            v, xy = mi.get_vertices_on_phys_tag(modelName, dim, physTag)
            mesh.setVertexGroupe(name, v)
        elif dim == 1:
            e2v = mi.get_toplevel_elements_vertices_on_phys_tag(modelName, dim, physTag)['edge']
            mesh.setEdgeGroupeFromEdge2VertexId(name, e2v)
        elif dim == 2:
            t2v = mi.get_toplevel_elements_vertices_on_phys_tag(modelName, dim, physTag)['tri']
            mesh.setTriGroupeFromTri2VertexId(name, t2v)
    return mesh 
        

class meshModification():
    def __init__(self, xy, tris):
        self.dim = 2
        self.xy = xy
        self.tris =  tris
        self.index_dtype = self.tris.dtype
        self.coord_dtype = self.xy.dtype
        self.deleted_tris    = np.zeros(0, dtype = np.int64 )
        #self.new_tris        = np.zeros((0,3), dtype = self.index_dtype)
        self.new_tris        = np.zeros((0,3), dtype = np.int64)
        self.xy_new_vertices = np.zeros((0,self.dim), dtype = self.coord_dtype)
        self.new_tris_parent = np.zeros(0, dtype = self.index_dtype)
        self.duplicated_vertices = np.zeros(0, dtype = self.index_dtype)
        self.split_edge_dtype = np.dtype( [('v0', self.index_dtype), ('v1', self.index_dtype), ('s', self.coord_dtype), ('vid', self.index_dtype)])
        self.split_edges = np.zeros(0, dtype = self.split_edge_dtype)
    def getNewMesh(self, verticesdata_oldmesh = None, verticesdata_modifyedmesh = None):
        xy   = np.vstack((self.xy, self.xy_new_vertices[len(self.duplicated_vertices):]))
        v2cut_added = np.hstack((self.duplicated_vertices, np.arange(len(self.xy), len(self.xy) + len(self.xy_new_vertices) - len(self.duplicated_vertices) ).astype(self.index_dtype) ))
        tris = np.vstack((np.delete(self.tris, self.deleted_tris, axis=0), v2cut_added[self.new_tris])).astype(int)
        if verticesdata_oldmesh is None :
            return xy, tris
        verticesdata = np.vstack((verticesdata_oldmesh[:, np.newaxis], verticesdata_modifyedmesh[len(self.duplicated_vertices):, np.newaxis] )).squeeze()
        return xy, tris, verticesdata
    def parents2sons(self):
        son2parent = self.new_tris_parent
        sons = np.argsort(son2parent)
        index_parent2sons = np.cumsum(np.bincount(son2parent[sons]+1, minlength=len(self.tris)+1))
        return index_parent2sons, sons

class dmesh():
    ''' A class to represent dynamic simplexes mesh in 2D '''
    class vertex():
        def __init__(self, vid, xy):
            self.id, self.xy, self.edges = vid, xy, set()
        def __str__(self):
            return 'id: '+str(self.id)+', xy :'+str(self.xy)
    class edge():
        def __init__(self, eid, vertices):
            self.id, (self.v0, self.v1), self.faces  = eid, vertices, set()
        def __str__(self):
            return 'id: '+str(self.id)+', vertices :'+str(self.v0.id) + ' ' + str(self.v1.id)
    class face():
        def __init__(self, fid, edges):
            self.id, (self.e0, self.e1, self.e2) = fid, edges    
        def __str__(self):
            return 'id: '+str(self.id)+', edges :'+str(self.e0.id) + ' ' + str(self.e1.id) + ' ' +  str(self.e2.id)
        def getVerticesId(self):
            v0, v1, v2 = self.getVertices()
            return [ v0.id, v1.id, v2.id ]
        def getVertices(self):
            e0, e2 = self.e0, self.e2
            if e2.v1 == e0.v0 : return [e0.v0, e0.v1, e2.v0]
            if e2.v1 == e0.v1 : return [e0.v1, e0.v0, e2.v0]
            if e2.v0 == e0.v0 : return [e0.v0, e0.v1, e2.v1]
            if e2.v0 == e0.v1 : return [e0.v1, e0.v0, e2.v1]
            raise
    def __init__(self, xy, tris): 
        e2v, t2e, __  = createEdgesFromTris(tris)
        v2eindex, v2e = upwardAdjacencies(e2v)
        e2tindex, e2t = upwardAdjacencies(t2e)
        self.vertices =  np.fromiter( (dmesh.vertex(vid, vxy) for vid, vxy in enumerate(xy)), count = len(xy) , dtype = object)
        self.edges    =  np.fromiter( (dmesh.edge(eid, self.vertices[vertices]) for eid, vertices in enumerate(e2v) ), count = len(e2v), dtype = object )
        self.faces    =  np.fromiter( (dmesh.face(fid, self.edges[edges]) for fid, edges in enumerate (t2e)), count = len(t2e), dtype = object )
        for v, edges in zip(self.vertices, (set( self.edges[v2e[s:e]] ) for (s,e) in zip(v2eindex[:-1], v2eindex[1:]) )) : v.edges = edges
        for e, faces in zip(self.edges, (set( self.faces[e2t[s:e]]) for (s,e) in zip(e2tindex[:-1], e2tindex[1:]) )) : e.faces = faces
        self.vertices = dict(enumerate(self.vertices))
        self.edges    = dict(enumerate(self.edges))
        self.faces    = dict(enumerate(self.faces))
        self.vid      = count(len(self.vertices))
        self.eid      = count(len(self.edges))
        self.fid      = count(len(self.faces))
    def getCommonVertex(e0, e1):
        v0e0, v1e0  = e0.v0, e0.v1
        v0e1, v1e1  = e1.v0, e1.v1
        if v0e0 == v0e1 : return v0e0
        if v0e0 == v1e1 : return v0e0
        if v1e0 == v0e1 : return v1e0
        if v1e0 == v1e1 : return v1e0
        raise
    def getCommonEdge(f0, f1):
        e = set((f0.e0, f0.e1, f0.e2))&set((f1.e0, f1.e1, f1.e2))
        if len(e) == 1 : return e.pop()
        raise
    def getCoordAndConnectivity(self):
        xy   = np.fromiter( (v.xy for v in self.vertices.values()), count = len(self.vertices) , dtype = (float, 2 ))
        tris = np.fromiter( (f.getVerticesId() for f in self.faces.values()), count = len(self.faces), dtype = (int, 3) )
        #e2e0 = np.fromiter( ( [f.e2.v0.id, f.e2.v1.id, f.e0.v0.id, f.e0.v1.id] for f in self.faces.values()), count = len(self.faces), dtype = (int, 4) )
        return xy, tris
    def insert_vertex(self, xyz):
        vid = next(self.vid)
        vn = dmesh.vertex(vid, xyz)
        self.vertices[vid] = vn
        return vn
    def insert_edge(self, evertices):
        eid = next(self.eid)
        en  = dmesh.edge(eid, evertices)
        for v in evertices : v.edges.add(en)
        self.edges[eid] = en
        return en
    def insert_face(self, fedges):
        fid = next(self.fid)
        fn  = dmesh.face(fid, fedges)
        for e in fedges : e.faces.add(fn)
        self.faces[fid] = fn
        return fn
    def _delete_face(self, f):
        for e in [f.e0, f.e1, f.e2] : e.faces.remove(f)
        del self.faces[f.id]
    def _delete_edge(self, e):
        if len(e.faces) > 0  : raise
        for v in [e.v0, e.v1] :  v.edges.remove(e)
        del self.edges[e.id]
    def _del_vertex(self, v):
        if len(v.edges) > 0  : raise
        del self.vertices[v.id] 
    def getVertices(self): return self.vertices.values()    
    def getEdges(self):    return self.edges.values()    
    def getFaces(self):    return self.faces.values()    
    def edgeCutCallBackPass(e, e0, e1): pass
    def faceCutCallBackPass(f, f0, f1): pass
    def split_edge(self, e, xyz, edgeCutCallBack = edgeCutCallBackPass , faceCutCallBack = faceCutCallBackPass):
        v0, v1 = e.v0, e.v1
        vn = self.insert_vertex(xyz)
        e0 = self.insert_edge([v0, vn])
        e1 = self.insert_edge([v1, vn])
        edgeCutCallBack(e, e0, e1)
        ftodel = [f for f in e.faces]
        for f in e.faces:
             fedges = [f.e0, f.e1, f.e2]
             fie = 0 if f.e0==e else  1 if f.e1 ==e else 2
             fe0, fe1, fe2 = fedges[fie], fedges[(fie+1)%3], fedges[(fie+2)%3]
             fv0 = dmesh.getCommonVertex(fe2, fe0)
             vc =  dmesh.getCommonVertex(fe1, fe2)
             fen = self.insert_edge([vn, vc])
             if fv0 == v0: f0e, f1e = [e0, fen, fe2], [e1, fe1, fen]
             else :        f0e, f1e = [e1, fen, fe2], [e0, fe1, fen]
             f0 = self.insert_face(f0e)
             f1 = self.insert_face(f1e)
             faceCutCallBack(f, f0, f1)
        for f in ftodel :self._delete_face(f)
        self._delete_edge(e)
        return vn

           
    
if __name__ =='__main__' :      
    import gmsh
    import gmsh_mesh_generator as mg
    import gmsh_post_pro_interface as gp
    import gmsh_mesh_model_interface as meshInterface
    import cProfile    
    gmsh.initialize()             
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)    
    
    testsmesh       = False
    testgmsh2smesh  = True
    dotopofuntest   = False
    dosmalltest     = False
    dolargetest     = False
    testPush2Edge   = False
    testrenumbering = True
    
    largetestelemsize = 0.05
    
    dopopup = False
    
    if testPush2Edge :
        pass
#        e2v, t2e, t2eflip = simplex.createEdgesFromTris(tris)
#        trisId       = np.arange(len(tris))
#        push2edge = pushTriData2EdgeOp(len(e2v), t2e, t2eflip)
#        trisIdOnEdges = push2edge.dot(trisId).reshape((-1,2))
#        gp.listPlotFieldLine(xy[e2v], trisIdOnEdges[:,0],  P0=True, viewname='EdgeLeftTriId',  IntervalsType='Numeric values')
#        gp.listPlotFieldLine(xy[e2v], trisIdOnEdges[:,1],  P0=True, viewname='EdgeRightTriId', IntervalsType='Numeric values')
#        gp.listPlotFieldTri(shrinktrisxy, trisId,          P0=True, viewname='TriId',          IntervalsType='Numeric values')
#        e2tStart, e2t = simplex.upwardAdjacencies(t2e)
#        boundaryEdgesIds =  np.where( (e2tStart[1:] - e2tStart[:-1]) == 1)[0]
#        for i in range(len(e2tStart)-1):
#            print(i, ' : ', e2t[e2tStart[i]:e2tStart[i+1] ])
#        gp.listPlotFieldLine(xy[e2v],  np.arange(len(e2v)),  P0 = True, viewname ='EdgeIds', IntervalsType='Numeric values', LineType = "3D cylinder", LineWidth = 4, VectorType = "Displacement", DisplacementFactor = 0., NbIso = 20)
#
#        gp.listPlotFieldLine(xy[e2v[boundaryEdgesIds]], boundaryEdgesIds,  P0 = True, viewname ='BoundEdgeIds', IntervalsType='Numeric values', LineType = "3D cylinder", LineWidth = 4, VectorType = "Displacement", DisplacementFactor = 0., NbIso = 20)

    if testsmesh :
        xy   = np.array([[0.,0.],[1,0.], [1.,1.], [0.,1.]])
        tris = np.array([[0,1,2],[2,3,0]])
        mesh = sMesh(xy, tris)
        print('tris',      mesh.getTris2Vertices())
        print('edge',      mesh.getEdges2Vertices())
        print('tris2edge', mesh.getTris2Edges()[0], '\n', mesh.getTris2Edges()[1] )
        print('find edge', mesh.findEdgeIds(np.array([[3,2]])))
    if testgmsh2smesh :
        import gmsh
        import gmsh_mesh_generator as mg
        modelName, physical_dict = mg.l_shape(relh = 1.)
        mesh = gmshModel2sMesh(modelName, groupeNames = ['dOmega_left', 'dOmega_right', 'dOmega_other', 'Omega', 'Omega1', 'Omega2', 'Omega3'])
        print(mesh.getVerticesCoord())
        print(mesh.getTris2Vertices())
        print(mesh.getEdges2Vertices())
        print(mesh.getTris2Edges()[0], '\n', mesh.getTris2Edges()[1])
        lE = mesh.getEdgesLength()
        for name in  mesh.getEdgeGroupeNames() :
            eids, eflip = mesh.getEdgeGroupe(name)
            print('Edge Groupe ', name, ' :')
            print(eids)
            print(eflip)
            normals = mesh.getEdgesNormal(eids, eflip)
            length =  mesh.getEdgesLength(eids)
            gp.listPlotFieldLine(mesh.getEdges2VerticesCoord(eids), normals, viewname = name+ '_Normals', P0= True)
            gp.listPlotFieldLine(mesh.getEdges2VerticesCoord(eids), length, viewname = name+ '_Length', P0= True)
            
            
        print(mesh.getEdgesNormal())
        gvnorm = gp.listPlotFieldLine(mesh.getEdges2VerticesCoord(), mesh.getEdgesNormal(), viewname = 'All Normals', P0= True)
        gvlen = gp.listPlotFieldLine(mesh.getEdges2VerticesCoord(), mesh.getEdgesLength(), viewname = 'All Length', P0= True)
        
        test_field = np.zeros(mesh.getNbTris())
        triGroupeNames = [i for i in physical_dict.keys() if i[0]=='O']
        for i,name in enumerate(triGroupeNames):
            tids = mesh.getTriangleGroupe(name)
            print('Triangle Groupe ', name, ' :')
            print(tids)
            test_field[tids] = i
        gvtest = gp.listPlotFieldTri(mesh.getTris2VerticesCoord(), test_field, viewname = 'Triangle_Test', P0= True)
        
        
        
       
            
        
            
        
        
        
    if dotopofuntest :
        f2v      = np.array([[0,1,2],[2,1,3]])
        e2v, f2e, __   =  createEdgesFromTris(f2v)
        v2e_index, v2e =  upwardAdjacencies(e2v)
        e2f_index, e2f =  upwardAdjacencies(f2e)
        v2v_index, v2v = _vertices2vertices(e2v, v2e_index, v2e)
        print('face 2 vertices')
        for f, fv in enumerate(f2v) : print(f, fv)
        print('edge 2 vertices')
        for e, ev in enumerate(e2v) : print(e, ev)
        print('vertex 2 edges')
        for  v, (start_ve, end_ve) in enumerate(zip(v2e_index[:-1], v2e_index[1:])) : print(v, v2e[start_ve:end_ve])
        print('face 2 edges')
        for f, fe in enumerate(f2e) : print(f, fe)
        print('edge 2 faces')
        for  e, (start_ef, end_ef) in enumerate(zip(e2f_index[:-1], e2f_index[1:])) : print(e, e2f[start_ef:end_ef])
        print('vertex 2 vertices (firstneighbor)')
        for  v, (start_vv, end_vv) in enumerate(zip(v2v_index[:-1], v2v_index[1:])) : print(v, v2v[start_vv:end_vv])
    if dosmalltest :
        print('test on mesh with 2 elements:')
        xy  = np.array([[0.,0.],[1,0.],[0.,1],[1,1]])
        tris = np.array([[0,1,2],[2,1,3]])
        print('create dynamic mesh')
        m = dmesh(xy, tris)
        xy2, tris2 = m.getCoordAndConnectivity()
        print(xy2)
        print(tris2)
        gp.listPlotFieldTri(xy2[tris2], np.ones(len(tris2)),  P0 = True,  viewname = "smallTest_meshLoaded")
        e = m.edges[2]
        m.split_edge(e, [.5,.5])
        xy3, tris3 = m.getCoordAndConnectivity()
        gp.listPlotFieldTri(xy3[tris3], np.ones(len(tris3)),  P0 = True,  viewname = "smallTest_oneEdgeSplit")        
    if dolargetest : 
        pr = cProfile.Profile()
        print('test on a gmsh mesh')
        modelName, physical_dict = mg.l_shape(largetestelemsize)
        xy   = meshInterface.get_vertices_coord(modelName)
        tris = meshInterface.get_triangles(modelName)
        print('number of elements:', len(tris))
        pr.enable()
        m = dmesh(xy, tris)
        xy2, tris2 = m.getCoordAndConnectivity()
        pr.disable()
        gp.listPlotFieldTri(xy2[tris2], np.ones(len(tris2)),  P0 = True,  viewname = "largeTest")
        pr.print_stats(sort='cumulative')
    if testrenumbering :
        modelName, physical_dict = mg.l_shape(0.3)
        mesh = gmshModel2sMesh(modelName)
        
        tris   = mesh.getTris2Vertices()
        tris  = tris[tris.shape[0]//4: 2*tris.shape[0]//4]
        newid = np.unique(tris.flatten())
        xy     = mesh.getVerticesCoord()
        xy2     = xy[newid]
        tris2  = np.searchsorted(newid, tris, side='left')
        
        gp.listPlotFieldTri(xy[tris], np.ones(len(tris)),  P0 = True,  viewname = "partial mesh")
        gp.listPlotFieldTri(xy[tris], tris, viewname = "partial mesh", IntervalsType = "Numeric values")
        
        gp.listPlotFieldTri(xy2[tris2], np.ones(len(tris2)),  P0 = True,  viewname = "partial mesh renumbered")
        gp.listPlotFieldTri(xy2[tris2], tris2, viewname = "partial mesh", IntervalsType = "Numeric values")
        
    if dopopup : gmsh.fltk.run()
    else:
        gmsh.write("test_oneLevelSimplexMesh.msh")
        gmsh.view.write(gvnorm, "test_oneLevelSimplexMesh.pos")
        gmsh.view.write(gvlen, "test_oneLevelSimplexMesh.pos", append=True)
        gmsh.view.write(gvtest, "test_oneLevelSimplexMesh.pos", append=True)
    gmsh.finalize()
