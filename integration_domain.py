'''
 # @ Author: hdandin
 # @ Created on: 2024-10-11 16:31:14
 # @ Modified time: 2024-10-11 16:31:31

Classes for line integration domains

'''

import numpy as np
import oneLevelSimplexMesh as sm

class PseudoContour:
    """ Class for definition of a pseudo contour not based on sMesh """

    def __init__(self, line_edges, line_vertices, edge_length, edge_normal, edge_cog, e2v, vxy,
                 tri_integpts=None):
        """
        *Args:
            - line_edges : ids of edges in contour (global mesh referencing)
            - line_vertices : ids of vertices in line_edges (global)
            - edge_length, edge_normal, edge_cog: edge characteristics
            - e2v : connectivity edge -> vertices (global)
            - vxy : vertex coordinates
            - tri_integpts : coordinates of mesh integration points (all mesh)

        All methods take eids as local indices in self.edges.
        """
        self.edges = line_edges
        self.vertices = line_vertices
        self.lengths = edge_length
        self.normals = edge_normal
        self.cogs = edge_cog
        self.e2v = e2v
        self.vxy = vxy
        self.e2vxy = vxy[e2v]
        if tri_integpts is not None:
            self.tri_integpts = tri_integpts
        else:
            self.tri_integpts = np.tile(edge_cog, (2,1))

    def get_nb_edges(self):
        """ Returns the number of edges """
        return len(self.edges)
    
    def get_vertex_coords(self, eids=None):
        """ Returns vertex coordinates """
        if eids is not None:
            return self.e2vxy[eids].reshape((-1,2,2))
        return self.e2vxy

    def get_lengths(self, eids=None):
        """ Returns edge lengths """
        if eids is not None:
            return self.lengths[eids]
        return self.lengths
    
    def get_cogs(self, eids=None):
        """ Returns edge cogs """
        if eids is not None:
            return self.cogs[eids].reshape((-1,2))
        return self.cogs

    def get_normals(self, eids=None):
        """ Returns outward normals """
        if eids is not None:
            return self.normals[eids].reshape((-1,2))
        return self.normals
    
    def get_displacements(self, U, eids=None):
        """ Returns nodal displacements of contour edges """
        u_left = U[self.e2v[:,0]]
        u_right = U[self.e2v[:,1]]
        if eids is not None:
            return np.stack((u_left[eids].reshape((-1,2)), u_right[eids].reshape((-1,2))), axis=1)
        return np.stack((u_left, u_right), axis=1)
    
    def get_stresses(self, stress, eids=None):
        """ Returns stresses left and right of contour edges """
        if eids is not None:
            return np.stack((stress[eids].reshape((-1,2,2)),
                             stress[eids].reshape((-1,2,2))), axis=1)
        return np.stack((stress, stress), axis=1)

    def get_tri_sizes(self, eids=None):
        """ Returns areas of triangles left and right of contour edges """
        if eids is not None:
            if np.isscalar(eids):
                return np.ones((1,2))
            return np.ones((len(eids),2))
        return np.ones((self.get_nb_edges(),2))
    
    def get_mesh_integpts(self, tids=None):
        """ Returns triangle integration point """
        if tids is not None:
            return self.tri_integpts[tids].reshape((-1,2))
        return self.tri_integpts

class Contour:
    """ Class for extraction of edges from mesh and QoI called by LineIntegration """

    def __init__(self, line_edges:np.ndarray, line_vertices:np.ndarray, mesh:sm.sMesh,
                 trisuv:np.ndarray):
        """
        *Args:
            - line_edges : ids of edges in contour (global mesh referencing)
            - line_vertices : ids of vertices in line_edges (global)
            - mesh : sMesh object
            - trisuv : coordinates of mesh integration points (all mesh)

        All methods take eids as local indices in self.edges.
        """
        self.edges = line_edges
        self.vertices = line_vertices
        self.mesh = mesh
        self.tri_integpts = trisuv
        self.e2t_contour = self.__get_tris() # connectivity edge -> triangles
        self.tris_lr = self.__get_tris_left_right() # triangles left and right of edges

    def __get_tris(self):
        """ Returns edge to triangle table for contour edges """
        e2t_r, e2t = sm.upwardAdjacencies(self.mesh.t2e)
        idx = np.vstack((e2t_r[self.edges], e2t_r[self.edges] + 1)).flatten('F')
        e2t_contour = e2t[idx]
        return e2t_contour

    def __get_tris_left_right(self):
        """ Returns mask for triangles on the left and right of contour edges """
        # get triangles between crack tip and contour
        _, _, tris_inside = np.intersect1d(self.mesh.getTriangleGroupe('contour_area'),
                                           self.e2t_contour,
                                           assume_unique=True,
                                           return_indices=True)
        # create mask for left triangles (~mask for right triangles)
        mask = np.ones(len(self.e2t_contour), dtype=bool)
        mask[tris_inside] = False
        return mask
    
    def get_nb_edges(self):
        """ Returns the number of edges """
        return len(self.edges)
    
    def get_vertex_coords(self, loc_eids=None):
        """ Returns vertex coordinates """
        if loc_eids is not None:
            return self.mesh.xy[self.vertices[loc_eids]].reshape((-1,2,2))
        return self.mesh.xy[self.vertices]
    
    def get_lengths(self, loc_eids=None):
        """ Returns edge lengths """
        if loc_eids is not None:
            return self.mesh.getEdgesLength(self.edges[loc_eids])
        return self.mesh.getEdgesLength(self.edges)
    
    def get_cogs(self, loc_eids=None):
        """ Returns edge cogs """
        if loc_eids is not None:
            return self.mesh.getEdgesCog(self.edges[loc_eids]).reshape((-1,2))
        return self.mesh.getEdgesCog(self.edges)

    def get_normals(self, loc_eids=None):
        """ Returns outward normals """
        edge_normal = self.mesh.getEdgesNormal(self.edges)
        directions = np.einsum('ij,ij->i', self.get_cogs(), edge_normal)
        edge_out_normal = np.where((directions > 0)[:, np.newaxis], edge_normal, -1.*edge_normal)
        if loc_eids is not None:
            return edge_out_normal[loc_eids].reshape((-1,2))
        return edge_out_normal
    
    def get_displacements(self, U, loc_eids=None):
        """ Returns nodal displacements of contour edges: shape = (eid,left/right,U) """
        u_left = U[self.mesh.e2v[:,0]]
        u_right = U[self.mesh.e2v[:,1]]
        if loc_eids is not None:
            return np.stack((u_left[loc_eids].reshape((-1,2)), u_right[loc_eids].reshape((-1,2))), axis=1)
        return np.stack((u_left, u_right), axis=1)
    
    def get_stresses(self, stress, loc_eids=None):
        """ Returns stresses left and right of contour edges: shape = (eid,left/right,stress) """
        stress_left = stress[self.e2t_contour[~self.tris_lr]]
        stress_right = stress[self.e2t_contour[self.tris_lr]]
        if loc_eids is not None:
            return np.stack((stress_left[loc_eids].reshape((-1,2,2)),
                             stress_right[loc_eids].reshape((-1,2,2))), axis=1)
        return np.stack((stress_left, stress_right), axis=1)
    
    def _getMeshTrisArea(self, tids = None, *, save = False):
        if hasattr(self, 'tarea') : tarea = self.tarea
        else :
             t2vxy = np.pad(self.mesh.getTris2VerticesCoord(save = save), ((0,0),(0,0),(0,1)),
                            'constant', constant_values=(0))
             tarea = 0.5 * np.linalg.norm( 
                 np.cross((t2vxy[:, 1, :] - t2vxy[:, 0, :]) , (t2vxy[:, 2, :] - t2vxy[:, 0, :]),
                          axis = 1), 
                 axis = 1)
        if tids is None: return tarea
        return tarea[tids]
    
    def get_tri_areas(self, loc_eids=None):
        """ Returns areas of triangles left and right of contour edges: 
        shape = (eid,left/right,area) """
        tri_area = self._getMeshTrisArea()
        area_left = tri_area[self.e2t_contour[~self.tris_lr]]
        area_right = tri_area[self.e2t_contour[self.tris_lr]]
        if loc_eids is not None:
            return np.stack((area_left[loc_eids].reshape((-1,1)),
                             area_right[loc_eids].reshape((-1,1))), axis=1)
        return np.stack((area_left, area_right), axis=1)

    def get_mesh_integpts(self, tids=None):
        """ Returns triangle integration point (tids = global indices) """
        if tids is not None:
            return self.tri_integpts[tids].reshape((-1,2))
        return self.tri_integpts


class Domain:
    """ Class for extraction of triangles from mesh and QoI called by DomainIntegration """

    def __init__(self, mesh:sm.sMesh, xy_g:np.ndarray, J:float, wg:np.ndarray, physical_tag:str=None):
        """
        *Args:
            - mesh : sMesh object
            - xy_g : coordinates of mesh integration points (all mesh)
            - J : Jacobian
            - wg : weights of integration points
            - physical_tag : gmsh tag for integration domain

        All methods take tids as local indices in self.triangles.
        """
        self.mesh = mesh
        if physical_tag is None:
            physical_tag = 'integration_domain'
        self.tris = mesh.getTriangleGroupe(physical_tag)
        self.vertices = mesh.getTris2Vertices()[self.tris]
        self.wg = wg
        self.integpts = self._get_local(xy_g)
        self.J = self._get_local(J)

    def _get_local(self, x):
        """ Returns x-values for tris in domain """
        if len(self.wg) > 1:
            raise ValueError('Not implemented')
        return x[self.tris]
    
    def get_nb_tris(self):
        """ Returns the number of triangles """
        return len(self.tris)
    
    def get_nb_integpts(self):
        """ Returns the number of integration points """
        return len(self.integpts)
    
    def get_vertex_coords(self, loc_tids=None):
        """ Returns vertex coordinates """
        if loc_tids is not None:
            return self.mesh.xy[self.vertices[loc_tids]].reshape((-1,3,2))
        return self.mesh.xy[self.vertices]
    
    def get_displacements(self, U, loc_tids=None):
        """ Returns nodal displacements of domain triangles: shape = (tid,3,U) """
        u_domain = U[self.mesh.t2v]
        if loc_tids is not None:
            return np.stack((u_domain[loc_tids,0].reshape((len(loc_tids),2)), 
                             u_domain[loc_tids,1].reshape((len(loc_tids),2)), 
                             u_domain[loc_tids,2].reshape((len(loc_tids),2))), axis=1)
        return np.stack((u_domain[:,0], u_domain[:,1], u_domain[:,2]), axis=1)
    
    def get_stresses(self, stress, loc_tids=None):
        """ Returns stresses at integration point: shape = (tid,1,stress) """
        if loc_tids is not None:
            return stress[self.tris[loc_tids]].reshape((len(loc_tids),1,4))
        return stress[self.tris].reshape((self.get_nb_tris(),1,4))
