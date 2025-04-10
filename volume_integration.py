'''
 # @ Author: hdandin
 # @ Created on: 2024-11-06 15:24:25
 # @ Modified time: 2024-11-06 15:24:28

 Bueckner-Chen volume integral
 '''

import numpy as np
import williams
from integration_domain import Domain

def virtual_displacement_ring(domain, xy):
    """ Virtual displacement field and derivative for ring domain
    q(r) = (rmax - r)/(rmax - rmin)
    """
    r, _ = williams.cartesian2polar(*domain.get_vertex_coords().T)
    rmax, rmin = np.max(r), np.min(r)

    q_xy = (rmax - np.sqrt(xy[:,0]**2 + xy[:,1]**2))/(rmax - rmin)
    dqdr = -1/(rmax - rmin)
    dqdx = dqdr * xy[:,0]/np.sqrt(xy[:,0]**2 + xy[:,1]**2)
    dqdy = dqdr * xy[:,1]/np.sqrt(xy[:,0]**2 + xy[:,1]**2)
    return q_xy, np.array([dqdx, dqdy]).T

class EquivalentDomainIntegral:
    """ Class for solving Bueckner-Chen EDI """

    def __init__(self, domain:Domain, kappa:float, mu:float, U:np.ndarray, stress:np.ndarray, 
                 grad_u:np.ndarray=None, strain:np.ndarray=None):
        """
        *Args :
            - contour : Contour object
            - kappa, mu : material paramters
            - U, stress : FE fields at integration points
            - strs_interp : method for stress interpolation from triangles to edges
            - method : integration method ("rect" = rectangle method, "exact" = exact integration
            on edges)
            - grad_u, strain : FE fields at integration points for J-integral
        """
        self.domain = domain
        self.kappa = kappa
        self.mu = mu
        self.U = U
        self.stress = stress

        self.grad_u = grad_u # for J-integral
        self.strain = strain

    def solve_bueckner(self, m: int, a_aux: float, b_aux: float):
        """ Solve Bueckner-Chen integral """
        
        # FE fields
        u_fe = self.U[self.domain.tris]
        stress_fe = self.stress[self.domain.tris]

        # auxiliary fields
        r, theta = williams.cartesian2polar(*self.domain.integpts.T)
        stress_aux = williams.get_stress(m, a_aux, b_aux, r, theta)
        u_aux = williams.get_displ(m, a_aux, b_aux, r, theta, self.kappa, self.mu)

        # virtual displacement function
        _, grad_q = virtual_displacement_ring(self.domain, self.domain.integpts)

        # integration
        term1 = (np.einsum('...ij,...i->...j',stress_aux,u_fe) 
                 - np.einsum('...ij,...i->...j',stress_fe,u_aux))
        integrand = np.einsum('...i,...i', term1, grad_q)
        brueckner = self.integrate(integrand)

        return brueckner

    def solve_j(self):
        """ Solve J-integral (Kuna, eq 6.48)
        J = - int (U * delta_kj - sigma_ij * u_i,k) q_k,j dV
        """

        # FE fields
        strain = self.strain[self.domain.tris]
        grad_u = self.grad_u[self.domain.tris]
        stress = self.stress[self.domain.tris]

        # stress work density
        U = 0.5 * np.einsum('...ij,...ij', stress, strain)

        # energy momentum

        # 1st term
        Q1 = U[...,np.newaxis,np.newaxis] * np.eye(2)

        # 2nd term
        Q2 = np.einsum('...ij,...ik->...kj', stress, grad_u)

        Q =  Q1 - Q2

        # virtual displacement gradient
        _, grad_q = virtual_displacement_ring(self.domain, self.domain.integpts)

        integrand = - np.einsum('...kj,...j->...k', Q, grad_q)

        J = self.integrate(integrand)

        return J
    
    def integrate(self, integrand: np.ndarray):
        """ Evaluate EDI with Gaussian quadrature """

        return np.dot(self.domain.wg * self.domain.J, integrand)
