'''
 # @ Author: hdandin
 # @ Created on: 2024-10-08 14:01:57
 # @ Modified time: 2024-10-08 14:02:32

Bueckner-Chen line integral

'''

import numpy as np
from scipy.integrate import quad
import williams
from integration_domain import Contour
import interpolation as ip

def cartesian2curve(xs, xyfield):
    """ interpolate field along curve
    *Args:
        - xs : cartesian coordinates of interpolation point,
        shape=(nb edges, field dimension)
        - xyfield : field to be interpolated in cartesian coordinates system,
        shape=(nb edges, nb nodes per edge, field dimension)
    Output:
        - s : curve coordinate (scalar),
        shape=(nb edges)
    """
    s =  xs/(xyfield[:,1,:] - xyfield[:,0,:]) - xyfield[:,0,:]
    return s

class LineIntegral:
    """ Class for solving Bueckner-Chen line integral
    Equation (20) in Melching & Breitbart (2023) """

    def __init__(self, contour:Contour, kappa:float, mu:float, U:np.ndarray, stress:np.ndarray,
                 fe_u_interp:'function', fe_stress_interp:'function', method:str, 
                 grad_u:np.ndarray=None, strain:np.ndarray=None):
        """
        *Args :
            - contour : Contour object
            - kappa, mu : material paramters
            - U, stress : FE fields
            - strs_interp : method for stress interpolation from triangles to edges
            - method : integration method ("rect" = rectangle method, "exact" = exact integration
            on edges)
            - grad_u, strain : FE fields for J-integral
        """
        self.contour = contour
        self.kappa = kappa
        self.mu = mu
        self.u_interp = fe_u_interp(contour, U)
        self.stress_interp = fe_stress_interp(contour, stress)
        self.method = method.lower()

        if grad_u is not None:
            self.grad_u_interp = fe_stress_interp(contour, grad_u)
            self.strain_interp = fe_stress_interp(contour, strain)

    def solve_bueckner(self, m: int, a_aux: float, b_aux: float) -> float:
        """ Solve Bueckner line integral with rectangle method, mid-point rule """
        
        # interpolated FE fields
        s = 0.5
        u_fe = self.u_interp.evaluate(s, self.contour.edges)
        stress_fe = self.stress_interp.evaluate(s, self.contour.edges)

        # auxiliary fields
        r, theta = williams.cartesian2polar(*self.contour.get_cogs().T)
        stress_aux = williams.get_stress(m, a_aux, b_aux, r, theta)
        u_aux = williams.get_displ(m, a_aux, b_aux, r, theta, self.kappa, self.mu)

        # integration
        term1 = (np.einsum('eij,ei->ej', stress_fe, u_aux) 
                 - np.einsum('eij,ei->ej', stress_aux, u_fe))
        integrand = np.einsum('ei,ei->e', term1, self.contour.get_normals())
        brueckner = np.dot(self.contour.get_lengths(), integrand)

        return brueckner
    
    def solve_j(self) -> float:
        """ Solve J-integral (Kuna, eq 6.48)
        J = int (U * delta_jk - sigma_ij * u_i,k) n_j ds
        """

        # FE fields
        s = 0.5
        strain = self.strain_interp.evaluate(s, self.contour.edges)
        grad_u = self.grad_u_interp.evaluate(s, self.contour.edges)
        stress = self.stress_interp.evaluate(s, self.contour.edges)

        # stress work density
        U = 0.5 * np.einsum('...ij,...ij', stress, strain)

        # energy momentum

        # 1st term
        Q1 = U[...,np.newaxis,np.newaxis] * np.eye(2)

        # 2nd term
        Q2 = np.einsum('...ij,...ik->...kj', stress, grad_u)

        Q =  Q1 - Q2

        integrand = np.einsum('...kj,...j->...k', Q, self.contour.get_normals())

        J = np.dot(self.contour.get_lengths(), integrand)

        return J
    
    def solve_bueckner_exactly(self, m:int, a_aux:float, b_aux:float) -> float:
        """ Exact integration over each edge """
        
        edge_integ = np.zeros(self.contour.get_nb_edges())
        for eid in range(self.contour.get_nb_edges()): # eid = local index
            edge_integ[eid] = self.integrate_bueckner_exactly_on_edge(eid, m, a_aux, b_aux)

        integral = np.dot(self.contour.get_lengths(), edge_integ)

        return integral
    
    def integrate_bueckner_exactly_on_edge(self, eid:int, m:int, a_aux:float, b_aux:float) -> tuple:
        """ Gaussian quadrature for exact integration along edge """
        integ, _ = quad(self.compute_integrand_bueckner, 0., 1., args=(eid, m, a_aux, b_aux))
        return integ
    
    def compute_integrand_bueckner(self, s:float, eid:int, m:int, a_aux:float, b_aux:float) -> float:
        """ Compute integrand
        
        *Args:
            - s : curve coordinate of Gauss integration point
            - eid : edge index (local)
            - m, a_aux, b_aux : order and coefficients for Williams auxiliary fields
        """

        u_fe, stress_fe = self.interpolate_fe_fields(s, eid)
        u_aux, stress_aux = self.compute_aux_fields(s, eid, m, a_aux, b_aux)
        
        integrand = np.einsum('ijk,ik->ij', stress_fe, u_aux)\
            - np.einsum('ijk,ik->ij', stress_aux, u_fe)
        integrand = np.einsum('ij,ij->i', integrand, self.contour.get_normals(eid))
        return integrand
    
    def interpolate_fe_fields(self, s:float, eid:int):
        """ Interpolate FE fields at Gauss integration points (linear ND interpolation) """
        u_s = self.u_interp.evaluate(s, self.contour.edges[eid])
        stress_s = self.stress_interp.evaluate(s, self.contour.edges[eid])
        return u_s, stress_s
    
    def compute_aux_fields(self, s:float, eid:int, m:int, a_aux:float, b_aux:float):
        """ Compute auxiliary fields (with Williams EEF) at Gaussian integration points """
        xys = ip.interp2curve(s, self.contour.get_vertex_coords(eid))
        r, theta = williams.cartesian2polar(*xys.T)
        stress_s = williams.get_stress(m, a_aux, b_aux, r, theta)
        u_s = williams.get_displ(m, a_aux, b_aux, r, theta, self.kappa, self.mu)
        return u_s, stress_s
    
    def solve_j_exactly(self) -> float:
        """ Exact integration over each edge """
        
        edge_integ = np.zeros(self.contour.get_nb_edges())
        for eid in range(self.contour.get_nb_edges()): # eid = local index
            edge_integ[eid] = self.integrate_j_exactly_on_edge(eid)

        integral = np.dot(self.contour.get_lengths(), edge_integ)

        return integral
    
    def integrate_j_exactly_on_edge(self, eid:int) -> tuple:
        """ Gaussian quadrature for exact integration along edge """
        integ, _ = quad(self.compute_integrand_j, 0., 1., args=eid)
        return integ

    def compute_integrand_j(self, s:float, eid:int):
        """ Compute integrand
        
        *Args:
            - s : curve coordinate of Gauss integration point
            - eid : edge index (local)
        """
        # FE fields
        strain = self.strain_interp.evaluate(s, self.contour.edges[eid])
        grad_u = self.grad_u_interp.evaluate(s, self.contour.edges[eid])
        stress = self.stress_interp.evaluate(s, self.contour.edges[eid])

        # stress work density
        U = 0.5 * np.einsum('...ij,...ij', stress, strain)

        # energy momentum

        # 1st term
        Q1 = U[...,np.newaxis,np.newaxis] * np.eye(2)

        # 2nd term
        Q2 = np.einsum('...ij,...ik->...kj', stress, grad_u)

        Q =  Q1 - Q2

        integrand = np.einsum('...kj,...j->...k', Q, self.contour.get_normals(eid))

        return integrand[0,0]
