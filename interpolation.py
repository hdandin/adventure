'''
 # @ Author: hdandin
 # @ Created on: 2024-10-11 16:29:55
 # @ Modified time: 2024-10-11 16:30:03

Interpolation methods for displacement and stress fields

 '''

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from integration_domain import Contour, Domain

def global2localeids(contour:Contour, eids:list):
    """ convert global (mesh) to local (contour) edge indices """
    return np.nonzero(contour.edges == eids)[0]

def interp2curve(s:float, xyfield:np.ndarray):
    """ interpolate field along curve """
    xys = xyfield[:,0,:] + s*(xyfield[:,1,:] - xyfield[:,0,:])
    return xys

class UInterpolatorNone:
    """ Pseudo-routine
    Interpolation method for computation of displacements (from FEM) on edge: no interpolation """
    def __init__(self, contour:Contour, U:np.ndarray):
        self.contour = contour
        self.U = U

    def evaluate(self, s:np.ndarray, eids:np.ndarray):
        """ Return displacement (if already known on edge) """
        if callable(self.U):
            return self.U(s)
        return self.U[eids].reshape((-1,2))
    
class UInterpolatorLinear:
    """ Interpolation method for computation of displacements (from FEM) on edge: linear
    interpolation
    
    *Args:
        - s : curve coordinate
        - eids : edge label (global indices)
    """
    def __init__(self, contour:Contour, U:np.ndarray):
        self.contour = contour
        self.U = U

    def evaluate(self, s:np.ndarray, eids:np.ndarray):
        """ Return displacement (if already known on edge) """
        u_e = self.contour.get_displacements(self.U, eids)
        return interp2curve(s, u_e).reshape((-1,2))
    
class StressInterpolatorNone:
    """ Pseudo-routine
    Interpolation method for computation of stress (from FEM) on edge: no interpolation """
    def __init__(self, contour:Contour, stress:np.ndarray):
        self.contour = contour
        self.stress = stress

    def evaluate(self, s:np.ndarray, eids:np.ndarray):
        """ Evaluate interpolation of displacement at points s """
        if callable(self.stress):
            return self.stress(s)
        return self.stress[eids].reshape((-1,2,2))

class StressInterpolatorNeighboursMean:
    """ Interpolation method for computation of stress (from FEM) on edge: mean value on edge from
    neighbouring integration points """
    def __init__(self, contour:Contour, stress:np.ndarray):
        self.contour = contour
        self.stress = self.contour.get_stresses(stress)

    def evaluate(self, s:np.ndarray, eids:np.ndarray):
        """ Evaluate interpolation of stress at points s from edge tids """
        loc_eids = global2localeids(self.contour, eids)
        return 0.5*(self.stress[loc_eids,0] + self.stress[loc_eids,1]).reshape((-1,2,2))
    
class StressInterpolatorLinearND:
    """ Interpolation method for computation of stress (from FEM) on edge: piecewise linear
    interpolation based on input data triangulation """
    def __init__(self, contour:Contour, stress:np.ndarray):
        self.contour = contour
        self.stress = stress
        self.interpolator = LinearNDInterpolator(contour.get_mesh_integpts(), stress)

    def evaluate(self, s:np.ndarray, eids:np.ndarray):
        """ Evaluate interpolation of stress at points s (eids=global indices) """
        loc_eids = global2localeids(self.contour, eids)
        xys = interp2curve(s, self.contour.get_vertex_coords(loc_eids)).reshape((-1,2))
        stress_interp = self.interpolator(xys[:,0], xys[:,1])
        return stress_interp.reshape((-1,2,2))
