'''
 # @ Author: hdandin
 # @ Created on: 2024-09-06 15:07:52
 # @ Modified time: 2024-09-06 15:10:24
 '''

import matplotlib.pyplot as plt
import numpy as np
import fracture_analysis_line as fa

cm = 1/2.54
figsz = (20*cm,15*cm)

marker_cycle = ['+','x','.','s','X','o']
R = 10. # contour radius
error_labels = ["integration","stress interpolation","FE discretisation"]
integ_labels = ["LI: ","EDI: "]


def rel_error(a, b):
    """ Relative error """
    return abs(a-b)/a

def create_figure(figsize, bot=0.1):
    """ Create figure with subplots (2,2) """
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=figsize)
    fig.subplots_adjust(bottom=bot,top=0.99,right=0.98,left=0.1,wspace=0.25,hspace=0.05)
    return fig, axs

def plot_williams(x, y, u, sigma, filename): 
    ''' Plot the VM stress and the displacement on the grid xs, ys for the williams serie'''

    plt.figure()
    plt.quiver(x, y, u[:,0], u[:,1], scale = 100.)
    plt.gca().axis('equal')
    plt.savefig(filename+'_displ.png')
    fig, ax = plt.subplots()
    vm = fa.vonmises(sigma)
    cb = plt.scatter(x, y, c=vm)
    plt.colorbar(cb)
    ax.axis('equal')
    fig.tight_layout()
    plt.savefig(filename+'_stress.png')
    plt.close()

def xytext_slope(arg):
    """ text position """
    return arg[0] + (arg[1]-arg[0])*3/4

def plot_cvge_relerror_log(errortype, folder=""):
    """ Plot convergence error
    x = h ; y = a_n
    """

    resfile = f"results/{folder}cvge_{errortype}.npz"
    npzfile = np.load(resfile)
    a_imp = npzfile['a_known']
    a_comp = npzfile['a_computed']
    element_lengths = npzfile['element_lengths']
    if 'a_1_from_j' in npzfile:
        a_1_from_j = npzfile['a_1_from_j']
    else:
        a_1_from_j = None

    tickers = []
    for i in range(len(a_imp)):
        str_i = str(i+1)
        tickers.append(r"$\left|(A_{stri}^W-A_{stri}\right)/A_{stri}^W|$".replace(
            'stri', str_i))

    integ_err = np.zeros((len(a_imp),len(element_lengths)))
    _, axs = create_figure(figsz)
    for n, ticker in enumerate(tickers):
        
        integ_err[n,:] = rel_error(a_imp[n], a_comp[n,:])
        axs[n//2,n%2].plot(np.log(element_lengths/R), np.log(integ_err[n,:]), 'o-', color='C0')

        # linear regression for convergence order
        p = np.polyfit(np.log(element_lengths/R), np.log(integ_err[n,:]), 1)
        xlim, ylim = axs[n//2,n%2].get_xlim(), axs[n//2,n%2].get_ylim()
        axs[n//2,n%2].text(xytext_slope(xlim), xytext_slope(ylim), f'slope: {p[0]:.2f}')

        if a_1_from_j is not None and n == 0:
            axs[n//2,n%2].plot(np.log(element_lengths/R), np.log(rel_error(a_imp[n], a_1_from_j[0,:])), 'o:', 
                    color='C0')

        if n//2 == 1:
            axs[n//2,n%2].set_xlabel(r'$\log(h/R)$')
        axs[n//2,n%2].set_ylabel(r"$\log($"+ticker+r"$)$")

    plt.savefig(f'results/cvge_relerrorlog_{errortype}.png', dpi=200)
    plt.close()
