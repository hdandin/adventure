#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:43:53 2024

@author: nchevaug
"""
import numpy as np

class Edge_gauss():
    def __init__(self, order=0):
        npt = np.int64(np.ceil((order+1)/2))
        s, w = np.polynomial.legendre.leggauss(npt)
        self.s = (1.+s)/2.
        self.w = w/2.
class Edge_nodes():
    def __init__(self, order=0):
        self.s = np.array([0.,1.])
        self.w = np.array([0.5,0.5])
class T3_edges():
    def __init__(self, edgeRule):
        s = edgeRule.s
        npt = len(s)
        u = np.hstack((s, 1-s, np.zeros(npt) ))
        v = np.hstack((np.zeros(npt), s, 1-s))
        self.uv = np.column_stack((u,v))
        self.w  = np.tile(edgeRule.w/3.,3)
class T3_gauss():
    _maxorder = 8
    def __init__(self, order=0, negweightOK = False):

        if order in [0,1] :
            self.w = np.array([0.5])
            self.uv = np.array([[1./3., 1./3.]])
            return
        if order == 2 :
            self.w = np.repeat(1./6.,3)   
            self.uv = np.array([[1./6., 1./6.], [2./3., 1./6.], [1./6, 2./3.] ])
            return
        if order == 3 and negweightOK :
           self.w = np.array([-9./32., 25./96., 25./96, 25/96. ])   
           self.uv = np.array([[1./3., 1./3.], [0.6, 0.2], [0.2,0.6], [0.2, 0.2] ])
           return
        if order in [3, 4]:
           self.w = np.repeat([ 0.109951743655322/2., 0.223381589678011/2.], 3)
           self.uv = np.array([
                    [0.816847572980459, 0.091576213509771], [0.091576213509771, 0.816847572980459],
                    [0.091576213509771, 0.091576213509771], [0.108103018168070, 0.445948490915965],
                    [0.445948490915965, 0.108103018168070], [0.445948490915965, 0.445948490915965]])
           return
        if order == 5 :
           self.w   = 0.5*np.array([0.225,  0.125939180544827, 0.125939180544827, 0.125939180544827, 0.132394152788506, 0.132394152788506, 0.132394152788506])
           self.uv  = np.array([
                   [0.333333333333333, 0.333333333333333], [0.797426985353087, 0.101286507323456],
                   [0.101286507323456, 0.797426985353087], [0.101286507323456, 0.101286507323456],
                   [0.470142064105115, 0.059715871789770], [0.059715871789770, 0.470142064105115],
                   [0.470142064105115, 0.470142064105115]])
           return
        if order == 6 :
           self.w   = 0.5*np.array([
                   0.050844906370207, 0.050844906370207, 0.050844906370207, 
                   0.116786275726379, 0.116786275726379, 0.116786275726379,
                   0.082851075618374, 0.082851075618374, 0.082851075618374, 
                   0.082851075618374, 0.082851075618374, 0.082851075618374])
           self.uv  = np.array([
                   [0.873821971016996, 0.063089014491502], [0.063089014491502, 0.873821971016996],
                   [0.063089014491502, 0.063089014491502], [0.501426509658179, 0.249286745170910],
                   [0.249286745170910, 0.501426509658179], [0.249286745170910, 0.249286745170910],
                   [0.636502499121399, 0.310352451033785], [0.310352451033785, 0.636502499121399],
                   [0.636502499121399, 0.053145049844816], [0.310352451033785, 0.053145049844816],
                   [0.053145049844816, 0.310352451033785], [0.053145049844816, 0.636502499121399]])
           return
        if order == 7 and negweightOK :
            self.w = np.array([ 
                -0.149570044467682, 0.175615257433208, 0.175615257433208,  
                 0.053347235608838, 0.053347235608838, 0.053347235608838,
                 0.077113760890257, 0.077113760890257, 0.077113760890257,
                 0.077113760890257, 0.077113760890257, 0.077113760890257])
            self.uv = np.array([
                    [0.333333333333333, 0.333333333333333], 
                    [0.479308067841920, 0.260345966079040], 
                    [0.260345966079040, 0.479308067841920],
                    [0.869739794195568, 0.065130102902216],
                    [0.065130102902216, 0.869739794195568],
                    [0.065130102902216, 0.065130102902216], 
                    [0.048690315425316, 0.312865496004874],
                    [0.312865496004874, 0.048690315425316],
                    [0.638444188569810, 0.048690315425316],
                    [0.048690315425316, 0.638444188569810],
                    [0.312865496004874, 0.638444188569810],
                    [0.638444188569810, 0.312865496004874],
                    ])
            return
        if order in [7,8] :
            self.w = np.array([
                0.144315607677787, 0.095091634267285, 0.095091634267285, 
                0.095091634267285, 0.103217370534718, 0.103217370534718, 
                0.103217370534718, 0.032458497623198, 0.032458497623198,
                0.032458497623198, 0.027230314174435, 0.027230314174435,
                0.027230314174435, 0.027230314174435, 0.027230314174435,
                0.027230314174435])
            self.uv = np.array([
                [0.333333333333333, 0.333333333333333], 
                [0.081414823414554, 0.459292588292723], 
                [0.459292588292723, 0.081414823414554],  
                [0.459292588292723, 0.459292588292723], 
                [0.658861384496480, 0.170569307751760],  
                [0.170569307751760, 0.658861384496480], 
                [0.170569307751760, 0.170569307751760], 
                [0.898905543365938, 0.050547228317031], 
                [0.050547228317031, 0.898905543365938], 
                [0.050547228317031, 0.050547228317031], 
                [0.008394777409958, 0.728492392955404], 
                [0.728492392955404, 0.008394777409958], 
                [0.263112829634638, 0.008394777409958], 
                [0.008394777409958, 0.263112829634638], 
                [0.263112829634638, 0.728492392955404],  
                [0.728492392955404, 0.263112829634638]])
            return
        raise  Exception('T3_gausspoints are only defined up to order {:d}'.format(T3_gauss._maxorder))
class T3_nodes():
    def __init__(self):
        self.w = np.repeat(0.5/3.,3)
        self.uv = np.array([[0.,0.], [1.,0.], [0.,1.]])
class T3_midedges():
    def __init__(self):
        self.w = np.repeat(1./6.,3)
        self.uv = np.array([[0.5, 0.], [0.5,0.5], [0.,0.5]])

if __name__== '__main__' :
    T0 = np.array([[0.,0.], [1.,0.], [0.,1.]])
    T1 = np.array([[1.,0.], [1.,1.], [0.,1.]])
    J = np.array([1.,1.])
    tris = np.array([T0, T1])
    fun = lambda xy : np.ones(xy.shape[:-1])
    fun = lambda xy : xy[...,0]**5
    for i in np.arange(6) :
        quad  = T3_gauss(i)
        u, v  = quad.uv[:,0], quad.uv[:,1] 
        N = np.array([(1.-u-v), u, v])
        egxy  = np.einsum ('esg, esd -> egd',  N[np.newaxis,:,:], tris) 
        egval = fun(egxy)
        integral = np.einsum('eg, eg', J[:, np.newaxis].dot(quad.w[np.newaxis,:]), egval ) 
        print('integration order :', i, 'integrale',  integral)
        
    import gmsh_post_pro_interface as gp
    import gmsh
   

    popUpGmsh  = True
    gmsh.initialize()
    gmsh.fltk.initialize()
    
    xy = np.array([[0.,0.],[1.,0.],[0.,1.]])
    
    edge   = np.array([[0,1]])
    xyedge = xy[edge]
    for order in range(9) :
        rule =  Edge_gauss(order)
        npt = len(rule.s)
        s = rule.s.reshape((npt,1))
        w = rule.w.reshape((npt,1))
        view  = gp.listPlotFieldPoint(s, w, viewname= 'Order = '+ str(order) , PointType='Scaled sphere', PointSize= 10) 
        gp.listPlotFieldLine(xyedge, np.array([0.001]), P0 = True, LineType='3D cylinder', LineWidth=1, gv = view)
    
    tri     = np.array([[0,1,2]])
    xyTri   = xy[tri]
    edges   = np.array([[0,1],[1,2],[2,0]])
    xyEdges = xy[edges] 

    for order in range(T3_gauss._maxorder+1) :
        print(order)
        rule =  T3_gauss(order)
        npt = len(rule.uv)
        uv = rule.uv.reshape((npt,2))
        w  = rule.w.reshape((npt,1))
        view  = gp.listPlotFieldPoint(uv, w, viewname= 'Order = '+ str(order) , PointType='Scaled sphere', PointSize= 10) 
        gp.listPlotFieldLine(xyEdges, np.array([0.001]*3), P0 = True, LineType='3D cylinder', LineWidth=1, gv = view)
        
    for order in range(T3_gauss._maxorder+1) :
        rule =  T3_edges(Edge_gauss(order))
        npt = len(rule.uv)
        uv = rule.uv.reshape((npt,2))
        w  = rule.w.reshape((npt,1))
        view  = gp.listPlotFieldPoint(uv, w, viewname= 'Order = '+ str(order) , PointType='3D sphere', PointSize= 10) 
        gp.listPlotFieldLine(xyEdges, np.array([0.001]*3), P0 = True, LineType='3D cylinder', LineWidth=1, gv = view)
 
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    #gmsh.open('post/quadrature.opt')
    #gp.gmshExportsViews('post/quadrature', 'mpg')
    #gp.gmshExportsViews('post/quadrature', 'pos')
    if popUpGmsh: 
        gmsh.fltk.run() 
    #gp.gmshExportsViews('post/quadrature', 'jpg') 
    #gmsh.write('post/quadrature.opt')
    
    gmsh.fltk.finalize()
    gmsh.finalize() 