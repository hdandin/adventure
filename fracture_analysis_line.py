'''
 # @ Author: hdandin
 # @ Created on: 2024-06-03 16:10:52
 # @ Modified time: 2024-07-24 17:25:58

Fracture analysis to extract Williams sub- and super-singular terms

'''

import numpy as np
import matplotlib.pyplot as plt
import williams
import line_integration as li

def lame_constants(E, nu, hyp2d):
    """ Return Lame constants (lambda, mu) from Young's modulus (E) and Poisson ration (nu) under 
    plane strain or plane stress approximation """
    if hyp2d == "plane strain":
        lamb, mu = (E*nu/(1.+nu)/(1.-2*nu), E/2./(1+nu))
    elif hyp2d == "plane stress":
        lamb, mu = (E*nu/(1.+nu)/(1.-nu), E/2./(1+nu))
    else:
        raise ValueError("Unknown 2d hypothesis:",hyp2d)
    return lamb, mu

def kolosov_constant(nu, hyp2d):
    """ Return Kolosv constant (kappa) from Poisson ratio (nu) under plane strain or plane stress
    approximation """
    if hyp2d == "plane strain":
        kappa = 3 - 4*nu
    elif hyp2d == "plane stress":
        kappa = (3 - nu)/(1 + nu)
    else:
        raise ValueError("Unknown 2d hypothesis:",hyp2d)
    return kappa

def vonmises(stress2d, sigmazz=0.):
    """ compute Von Mises equivalent stress """
    sxx = stress2d[:,0,0]
    syy = stress2d[:,1,1]
    sxy = stress2d[:,0,1]
    szz = sigmazz
    p   = (sxx+syy+szz)/3
    dxx = sxx-p
    dyy = syy-p
    dzz = szz-p
    dxy = sxy
    return np.sqrt(dyy**2+dxx**2+dzz**2+2*dxy**2)


class FractureAnalysisLine:
    """ Class for determination of Williams EEF subsingular terms 
    
    Methods:
        * run - solve for Brueckner-Chen integral
        * getWilliamsStress - reconstruct Williams stress field
        * getWilliamsU - reconstruct Williams displacement field
    """

    def __init__(self, contour: li.Contour, E: float, nu: float, williams_orders,
                 hyp2d: str="plane stress"):
        self.contour = contour
        self.E = E
        self.nu = nu
        self.lamb, self.mu = lame_constants(E, nu, hyp2d)
        self.kappa = kolosov_constant(nu, hyp2d)
        self.hyp2d = hyp2d
        self.williams_orders = williams_orders
        self.a_n = []
        self.b_n = []

    def run(self, U:np.ndarray, stress:np.ndarray, fe_u_interp:'function',
            fe_stress_interp:'function', integ_method:str="exact",
            grad_u:np.ndarray=None, strain:np.ndarray=None):
        """ Solve line integral to compute William's coefficients for given orders """

        integral = li.LineIntegral(self.contour, self.kappa, self.mu, U, stress, fe_u_interp,
                                      fe_stress_interp, integ_method, grad_u, strain)

        for n in self.williams_orders:
            coefs = self.williams_coefs_from_integral(integral, n, 1., 1.)
            self.a_n.append(coefs[0])
            self.b_n.append(coefs[1])

        if grad_u is not None:
            j = integral.solve_j()
            k_1 = (j[0]*self.E)**0.5
            a_1_from_j = k_1 / (2*np.pi)**0.5
            return a_1_from_j, j[0]

    def williams_coefs_from_integral(self, integral: li.LineIntegral, n:int, a_aux:float,
                                     b_aux:float):
        """ Compute Williams coefficients from Brueckner-Chen integral,
        Equation (27) in Melching & Breitbart (2023)

        """
        m = -n
        if integral.method == "exact":
            brueckner_i = integral.solve_bueckner_exactly(m, a_aux, 0.)
            brueckner_ii = integral.solve_bueckner_exactly(m, 0., b_aux)
        else:
            brueckner_i = integral.solve_bueckner(m, a_aux, 0.)
            brueckner_ii = integral.solve_bueckner(m, 0., b_aux)
        a_n = - self.mu/(self.kappa + 1) * 1/(np.pi*n*(-1)**(n+1)) * brueckner_i
        b_n = - self.mu/(self.kappa + 1) * 1/(np.pi*n*(-1)**(n+1)) * brueckner_ii

        return a_n, b_n

    def get_williams_stress(self, x, y, compute_sigmazz=False):
        """ compute Williams stress field """
        r, theta = williams.cartesian2polar(x, y)
        stress_w = np.zeros((*theta.shape,2,2))
        for n, a_n, b_n in zip(self.williams_orders, self.a_n, self.b_n):
            stress_w += williams.get_stress(n, a_n, b_n, r, theta)
        if not compute_sigmazz: 
            return stress_w
        if self.hyp2d == "plane strain":
            sigmazz  = self.lamb/(2*self.lamb + 2*self.mu) *(stress_w[:,0,0] + stress_w[:,1,1])
        else :
            sigmazz = np.zeros(x.shape)
        return stress_w, sigmazz
    
    def get_williams_displ(self, x, y):
        """ compute Williams displacement field """
        r, theta = williams.cartesian2polar(x,y)
        u_w = np.zeros((*theta.shape,2))
        for n, a_n, b_n in zip(self.williams_orders, self.a_n, self.b_n):
            u_w += williams.get_displ(n, a_n, b_n, r, theta, self.kappa, self.mu)
        return u_w
    
    def get_williams_grad_displ(self, x, y):
        """ compute gradient of Williams displacement field """
        grad_u_w = np.zeros((*x.shape,2,2))
        for n, a_n, b_n in zip(self.williams_orders, self.a_n, self.b_n):
            grad_u_w += williams.get_grad_displ(n, a_n, b_n, np.array([x,y]).T, self.kappa, self.mu)
        return grad_u_w
    
    def get_williams_strain(self, x, y):
        """ compute strain from Williams displacement field """
        grad_u_w = self.get_williams_grad_displ(x, y)
        strain_w = 0.5*(grad_u_w.swapaxes(-1,-2) + grad_u_w)
        return strain_w

def plot_fields(filename, x, y, u, xx=None, yy=None, sigma=None, scale=1.): 
    ''' Plot the displacement and VM stress '''

    plt.figure()
    plt.quiver(x, y, u[:,0], u[:,1], scale=scale, label=f"norm(u)={np.max(np.linalg.norm(u)):.4f}")
    plt.gca().axis('equal')
    plt.legend()
    plt.savefig(filename+'_displ.png')
    plt.close()
    if xx is None:
        return
    fig, ax = plt.subplots()
    vm = vonmises(sigma)
    cb = plt.scatter(xx, yy, c=vm)
    plt.colorbar(cb)
    ax.axis('equal')
    fig.tight_layout()
    plt.savefig(filename+'_stress.png')
    plt.close()
