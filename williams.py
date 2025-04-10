'''
 # @ Author: hdandin
 # @ Created on: 2024-06-03 15:49:12
 # @ Modified time: 2024-07-24 17:26:16
 '''

import unittest
import numpy as np
import sympy as sy

def cartesian2polar(x, y):
    """ Transforms cartesian coordinates into polar coordinates """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cartesian(r, theta):
    """ Transforms polar coordinates into cartesian coordinates """
    return np.array([r*np.cos(theta), r*np.sin(theta)]).T

def stress_eigenfunctions(n: int, theta:np.ndarray):
    """ n-th trigonometric eigenfunctions in polar coordinates for stresses """

    fi_xx = n/2 * ((2 + (-1)**n + n/2) * np.cos((n/2 - 1) * theta) 
                    - (n/2 - 1) * np.cos((n/2 - 3) * theta))
    fi_yy = n/2 * ((2 - (-1)**n - n/2) * np.cos((n/2 - 1) * theta) 
                    + (n/2 - 1) * np.cos((n/2 - 3) * theta))
    fi_xy = n/2 * ((n/2 - 1) * np.sin((n/2 - 3) * theta) 
                    - (n/2 + (-1)**n) * np.sin((n/2 - 1) * theta))
    fii_xx = n/2 * ((-2 + (-1)**n - n/2) * np.sin((n/2 - 1) * theta) 
                    + (n/2 - 1) * np.sin((n/2 - 3) * theta))
    fii_yy = n/2 * ((-2 - (-1)**n + n/2) * np.sin((n/2 - 1) * theta) 
                    - (n/2 - 1) * np.sin((n/2 - 3) * theta))
    fii_xy = n/2 * ((n/2 - 1) * np.cos((n/2 - 3) * theta) 
                    - (n/2 - (-1)**n) * np.cos((n/2 - 1) * theta))
    return fi_xx, fi_yy, fi_xy, fii_xx, fii_yy, fii_xy

def displ_eigenfunctions(n: int, theta:np.ndarray, kappa:float):
    """ n-th trigonometric eigenfunctions in polar coordinates for displacements """

    gi_x = (kappa + (-1)**n + n/2) * np.cos(n/2 * theta) - n/2 * np.cos((n/2 - 2) * theta)
    gi_y = (kappa - (-1)**n - n/2) * np.sin(n/2 * theta) + n/2 * np.sin((n/2 - 2) * theta)
    gii_x = (-kappa + (-1)**n - n/2) * np.sin(n/2 * theta) + n/2 * np.sin((n/2 - 2) * theta)
    gii_y = (kappa + (-1)**n - n/2) * np.cos(n/2*theta) + n/2 * np.cos((n/2 - 2) * theta)
    return gi_x, gi_y, gii_x, gii_y

def displ_eigenfunctions_dtheta(n: int, theta: np.ndarray, kappa:float):
    """ n-th gradients of trigonometric eigenfunctions in polar coordinates for displacements """
    
    dgi_x_dtheta = n/2 * ( - (kappa + (-1)**n + n/2) * np.sin(n/2 * theta) \
        + (n/2 - 2) * np.sin((n/2 - 2) * theta) )
    dgi_y_dtheta = n/2 * ( (kappa - (-1)**n - n/2) * np.cos(n/2 * theta) \
        + (n/2 - 2) * np.cos((n/2 - 2) * theta) )
    dgii_x_dtheta = n/2 * ( (-kappa + (-1)**n - n/2) * np.cos(n/2 * theta) \
        + (n/2 - 2) * np.cos((n/2 - 2) * theta) )
    dgii_y_dtheta = -n/2 * ( (kappa + (-1)**n - n/2) * np.sin(n/2*theta) \
        + (n/2 - 2) * np.sin((n/2 - 2) * theta) )
    return dgi_x_dtheta, dgi_y_dtheta, dgii_x_dtheta, dgii_y_dtheta

def get_stress(n: int, a_n: float, b_n: float, r:float, theta:np.ndarray):
    """ 2d stress tensor from Williams n-th order coefficient """

    fi_xx, fi_yy, fi_xy, fii_xx, fii_yy, fii_xy = stress_eigenfunctions(n, theta)
    if np.isscalar(theta):
        sig_n = np.zeros((1,2,2))
    else:
        sig_n = np.zeros((*theta.shape,2,2))
    sig_n[:,0,0] = r**(n/2 - 1) * (a_n * fi_xx + b_n * fii_xx)
    sig_n[:,1,1] = r**(n/2 - 1) * (a_n * fi_yy + b_n * fii_yy)
    sig_n[:,0,1] = r**(n/2 - 1) * (a_n * fi_xy + b_n * fii_xy)
    sig_n[:,1,0] = r**(n/2 - 1) * (a_n * fi_xy + b_n * fii_xy)
    return sig_n

def get_displ(n: int, a_n: float, b_n: float, r:float, theta:np.ndarray, kappa:float, mu:float):
    """ 2d displacement vector from Williams n-th order coefficient """

    gi_x, gi_y, gii_x, gii_y = displ_eigenfunctions(n, theta, kappa)
    if np.isscalar(theta):
        u_n = np.zeros((1,2))
    else:
        u_n = np.zeros((*theta.shape,2))
    u_n[:,0] = r**(n/2)/(2*mu) * (a_n * gi_x + b_n * gii_x)
    u_n[:,1] = r**(n/2)/(2*mu) * (a_n * gi_y + b_n * gii_y)
    return u_n

def get_grad_displ(n: int, a_n: float, b_n: float, xy:np.ndarray, kappa:float, mu:float):
    """ 2d gradient of displacement vector from Williams n-th order coefficient
    gradient in cartesian coordinates with composition of derivatives """

    r, theta = cartesian2polar(*xy.T)
    gi_x, gi_y, gii_x, gii_y = displ_eigenfunctions(n, theta, kappa)
    dgi_x_dtheta, dgi_y_dtheta, dgii_x_dtheta, dgii_y_dtheta = displ_eigenfunctions_dtheta(n, theta,
                                                                                           kappa)
    du_x_dr = n * r**(n/2 - 1) / (4 * mu) * (a_n * gi_x + b_n * gii_x) # du_x(r,theta)/dr
    du_y_dr = n * r**(n/2 - 1) / (4 * mu) * (a_n * gi_y + b_n * gii_y) # du_y(r,theta)/dr
    du_x_dtheta = r**(n/2) / (2 * mu)* (a_n * dgi_x_dtheta + b_n * dgii_x_dtheta)
    du_y_dtheta = r**(n/2) / (2 * mu)* (a_n * dgi_y_dtheta + b_n * dgii_y_dtheta)

    dr_dx = xy[:,0] / np.sqrt(xy[:,0]**2 + xy[:,1]**2) # x / (x^2 + y^2)^0.5
    dtheta_dx = - xy[:,1] / (xy[:,0]**2 + xy[:,1]**2) # -y / (x^2 + y^2)
    dr_dy = xy[:,1] / np.sqrt(xy[:,0]**2 + xy[:,1]**2) # y / (x^2 + y^2)^0.5
    dtheta_dy = xy[:,0] / (xy[:,0]**2 + xy[:,1]**2) # x / (x^2 + y^2)


    if np.isscalar(theta):
        grad_u_n = np.zeros((1,2,2))
    else:
        grad_u_n = np.zeros((*theta.shape,2,2))

    grad_u_n[:,0,0] = du_x_dr * dr_dx + du_x_dtheta * dtheta_dx
    grad_u_n[:,1,0] = du_y_dr * dr_dx + du_y_dtheta * dtheta_dx
    grad_u_n[:,0,1] = du_x_dr * dr_dy + du_x_dtheta * dtheta_dy
    grad_u_n[:,1,1] = du_y_dr * dr_dy + du_y_dtheta * dtheta_dy

    return grad_u_n

def get_grad_displ_cyl(n: int, a_n: float, b_n: float, r:float, theta:np.ndarray, kappa:float, 
                       mu:float):
    """ 2d gradient of displacement vector from Williams n-th order coefficient
    gradient in cylindric coordinates """

    gi_x, gi_y, gii_x, gii_y = displ_eigenfunctions(n, theta, kappa)
    dgi_x_dtheta, dgi_y_dtheta, dgii_x_dtheta, dgii_y_dtheta = displ_eigenfunctions_dtheta(n, theta,
                                                                                           kappa)

    u_n_cart = get_displ(n, a_n, b_n, r, theta, kappa, mu)

    if np.isscalar(theta):
        # u_n_pol = np.zeros((1,2))
        grad_u_n = np.zeros((1,2,2))
    else:
        # u_n_pol = np.zeros((*theta.shape,2))
        grad_u_n = np.zeros((*theta.shape,2,2))
    
    u_r = u_n_cart[:,0] * np.cos(theta) + u_n_cart[:,1] * np.sin(theta)
    u_theta = - u_n_cart[:,0] * np.sin(theta) + u_n_cart[:,1] * np.cos(theta)

    du_r_dr = n * r**(n/2 - 1) / (4*mu) * ( np.cos(theta) * (a_n * gi_x + b_n * gii_x) \
        + np.sin(theta) * (a_n * gi_y + b_n * gii_y) )
    du_theta_dr = n * r**(n/2 - 1) / (4*mu) * ( -np.sin(theta) * (a_n * gi_x + b_n * gii_x) \
        + np.cos(theta) * (a_n * gi_y + b_n * gii_y) )
    du_r_dtheta = r**(n/2) / (2*mu) * ( np.cos(theta) * ( a_n * (gi_y + dgi_x_dtheta) \
        + b_n * (gii_y + dgii_x_dtheta)) \
            + np.sin(theta) * ( a_n * (-gi_x + dgi_y_dtheta) \
                + b_n * (-gii_x + dgii_y_dtheta) ) )
    du_theta_dtheta = r**(n/2) / (2*mu) * ( np.cos(theta) * ( a_n * (-gi_x + dgi_y_dtheta) \
        + b_n * (-gii_x + dgii_y_dtheta)) \
            - np.sin(theta) * ( a_n * (gi_y + dgi_x_dtheta) \
                + b_n * (gii_y + dgii_x_dtheta) ) )

    # matrix in cylindric coordinates
    grad_u_n[:,0,0] = du_r_dr
    grad_u_n[:,1,0] = du_theta_dr
    grad_u_n[:,0,1] = 1/r * du_r_dtheta - u_theta/r
    grad_u_n[:,1,1] = 1/r * du_theta_dtheta + u_r/r

    # transform matrix into cartesian coordinates
    P = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    grad_u_n = np.einsum('ji,...jk,kl->...il', P, grad_u_n, P)

    return grad_u_n

def displ_eigenfunctions_sympy(n, kappa, theta):
    """ n-th trigonometric eigenfunctions in polar coordinates for displacements with sympy """

    gi_x = (kappa + (-1)**n + n/2) * sy.cos(n/2 * theta) - n/2 * sy.cos((n/2 - 2) * theta)
    gi_y = (kappa - (-1)**n - n/2) * sy.sin(n/2 * theta) + n/2 * sy.sin((n/2 - 2) * theta)
    gii_x = (-kappa + (-1)**n - n/2) * sy.sin(n/2 * theta) + n/2 * sy.sin((n/2 - 2) * theta)
    gii_y = (kappa + (-1)**n - n/2) * sy.cos(n/2*theta) + n/2 * sy.cos((n/2 - 2) * theta)
    
    return gi_x, gi_y, gii_x, gii_y

def grad_displ_cart_sympy(n, a_n, b_n, x, y, kappa, mu):
    """ 2d gradient of displacement vector from Williams n-th order coefficient
    gradient in cartesian coordinates with sympy """

    r = (x**2 + y**2)**sy.Rational(1,2)
    theta = sy.atan2(y, x)

    gi_x, gi_y, gii_x, gii_y = displ_eigenfunctions_sympy(n, kappa, theta)

    u_x = r**(sy.Rational(1,2)*n) * sy.Rational(1,2)/mu * (a_n * gi_x + b_n * gii_x)
    u_y = r**(sy.Rational(1,2)*n) * sy.Rational(1,2)/mu * (a_n * gi_y + b_n * gii_y)

    grad_u = sy.Matrix([[u_x.diff(x), u_x.diff(y)], [u_y.diff(x), u_y.diff(y)]])
    
    return grad_u

def grad_displ_cyl_sympy(n, a_n, b_n, r, theta, kappa, mu):
    """ 2d gradient of displacement vector from Williams n-th order coefficient
    gradient in cylindric coordinates with sympy """

    gi_x, gi_y, gii_x, gii_y = displ_eigenfunctions_sympy(n, kappa, theta)

    u_x = r**(sy.Rational(1,2)*n) * sy.Rational(1,2)/mu * (a_n * gi_x + b_n * gii_x)
    u_y = r**(sy.Rational(1,2)*n) * sy.Rational(1,2)/mu * (a_n * gi_y + b_n * gii_y)

    # u in cylindric coordinates
    u_r = u_x * sy.cos(theta) + u_y * sy.sin(theta)
    u_theta = -u_x * sy.sin(theta) + u_y * sy.cos(theta)

    # grad(u) in cylindric coordinates
    grad_u_rr = u_r.diff(r) # du_r/dr
    grad_u_thetar = u_theta.diff(r) # du_theta/dr
    grad_u_rtheta = 1/r * u_r.diff(theta) - u_theta/r # 1/r * du_r/dtheta - u_theta/r
    grad_u_thetatheta = 1/r * u_theta.diff(theta) + u_r/r # 1/r * du_theta/dtheta + u_r/r
    grad_u_cyl = sy.Matrix([[grad_u_rr, grad_u_rtheta], [grad_u_thetar, grad_u_thetatheta]])

    # grad(u) in cartesian coordinates
    P = sy.Matrix([[sy.cos(theta), sy.sin(theta)], [-sy.sin(theta), sy.cos(theta)]])
    grad_u_cart = P.T @ grad_u_cyl @ P
    
    return grad_u_cart

class TestWilliams(unittest.TestCase):
    """ unit test class Williams """
    def test_grad_displ_with_sympy(self):
        """ test displacement gradient with sympy """
        print('\n--- Test displacement gradient with sympy ---')

        n = sy.symbols("n", integer=True)
        kappa, mu = sy.symbols("kappa, mu", real=True)
        a_n, b_n = sy.symbols("a_n, b_n", real=True)
        r = sy.symbols("r", positive=True)
        theta = sy.symbols("theta", real=True)
        x, y = sy.symbols("x, y", real=True)

        E, nu = 1., 0.3
        K, MU = (3 - nu)/(1 + nu), E/2./(1+nu)
        N = 1
        A = 1.
        B = 0.
        R, TH = 1., np.pi/2
        XY = polar2cartesian(R, TH)

        grad_u_cart = grad_displ_cart_sympy(n, a_n, b_n, x, y, kappa, mu)
        true_cart = grad_u_cart.evalf(subs={n: N, a_n: A, b_n: B, x: XY[0], y: XY[1], kappa: K, mu: MU})

        grad_u_cyl = grad_displ_cyl_sympy(n, a_n, b_n, r, theta, kappa, mu)
        true_cyl = grad_u_cyl.evalf(subs={n: N, a_n: A, b_n: B, r: R, theta: TH, kappa: K, mu: MU})

        test_cart = get_grad_displ(N, A, B, XY[np.newaxis,:], K, MU)
        test_cyl = get_grad_displ_cyl(N, A, B, R, TH, K, MU)

        print('\ntrue cyl :',true_cyl)
        print('true cart :',true_cart)

        print('\ngrad_u_xx\ntrue cyl :',true_cyl[0,0])
        print('true cart :',true_cart[0,0])
        print('test cyl :',test_cyl[0,0,0])
        print('test cart :',test_cart[0,0,0])

        print('\ngrad_u_xy\ntrue cyl :',true_cyl[0,1])
        print('true cart :',true_cart[0,1])
        print('test cyl :',test_cyl[0,0,1])
        print('test cart :',test_cart[0,0,1])

        self.assertTrue(np.all(abs(true_cart - test_cart) < 1e-4), msg="cart sympy/numpy")
        self.assertTrue(np.all(abs(true_cyl - test_cyl) < 1e-4), msg="cyl sympy/numpy")
        self.assertTrue(np.all(abs(np.array((true_cyl - true_cart)).astype(np.float64)) < 1e-4), msg="sympy cart/cyl")
