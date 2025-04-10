#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 07:43:08 2024

@author: nchevaug
"""
from __future__ import annotations
from warnings import warn
import time
import unittest
import numpy as np
import scipy as sp
import sksparse.cholmod
import gmsh

import function_spaces as fs
import quadrature as quad
import sparse_tools as st
import oneLevelSimplexMesh as sm
import gmsh_post_pro_interface as gp
from execTools import Profile
import gmsh_mesh_generator as mg

class tensor2222CSR():
    ''' grouping CSR matrix repesentation of order 4 tensor with base space of dimension 2

        For now we have, for A a second order tensor (repesented as a 2d matrix in rowmajor storage)
            - ISym:   .5 (AT + A) = ISym: A  ->  ISym.dot(A.flatten()).resahpe((2, 2))
            - IxI      I*trace(A) = IxI: A    ->  IxI.dot(A.flatten()).resahpe((2, 2))
    '''
    _data = np.array([1., .5, .5, .5, .5, 1.])
    _row = np.array([0, 1, 2, 1, 2, 3])
    _col = np.array([0, 1, 1, 2, 2, 3])
    ISym = sp.sparse.csr_array((_data, (_row, _col)))
    _data = np.array([1., 1., 1., 1.])
    _col = np.array([0, 0, 3, 3])
    _row = np.array([0, 3, 0, 3])
    IxI = sp.sparse.csr_array((_data, (_row, _col)))

ISym = tensor2222CSR.ISym
IxI = tensor2222CSR.IxI

def YoungPoisson2LambdaNu(E, nu):
    """ Return Lame coefficients (Lambda, mu) from Young Modulus (E) and poisson ration nu """
    lamb, mu = (E*nu/(1.+nu)/(1.-2*nu), E/2./(1+nu))
    return lamb, mu

def planeStrainC2222CSR(lam, mu):
    ''' Compute the C operator (tensor 4) for isotropic material

    such as sigma = C:eps'''
    return lam*IxI + 2*mu*ISym

def getSpaceEntity(edgeGroupeNames, space):
    ''' get the space's edge id corresponding to the edgeGroupe Name'''
    edgeGroupeNames2EId = {}
    for name in edgeGroupeNames:
        edgeGroupeNames2EId[name] = space.m.getEdgeGroupe(name)[0]
    return edgeGroupeNames2EId

def getDofsAndInterpolationPoints(mesh, groupeNames, space):
    ''' get the space's degree of freedom id and the corresponding interpolation point

    corresponding to the groupeNames (alist of names). Usefull for Diichlet BC for example'''
    groupeName2DofId = {}
    groupeName2Dofxy = {}
    funName = getDofsAndInterpolationPoints.__name__
    for physName in groupeNames:
        eid, _ = mesh.getEdgeGroupe(physName)
        if eid.size == 0:
            vids = mesh.getVertexGroupe(physName)
            if vids.size == 0:
                warn('In '+funName+' no vertices found in groupe '+physName+'.')
                raise ValueError("Can't find any dofs attached to group", physName )
            if space.interpolatoryAt == 'midEdge':
                raise ValueError("No interpolation point associated to point for space \
                                 interpolatory at mid edge", physName )
            coord  = mesh.getVerticesCoord()[vids]
            dofsid = space.vertexid2dofid(vids)
            groupeName2DofId[physName] = dofsid.reshape((vids.size, -1))
            groupeName2Dofxy[physName] = coord.reshape((-1, 2))  
        elif space.interpolatoryAt == 'vertex':
            vertices = np.unique((mesh.getEdges2Vertices()[eid]).flatten())
            coord = mesh.getVerticesCoord()[vertices]
            dofsid = space.vertexid2dofid(vertices)
            groupeName2DofId[physName] = dofsid
            groupeName2Dofxy[physName] = coord.reshape((-1, 2))
        elif space.interpolatoryAt == 'midEdge':
            groupeName2DofId[physName] = space.edgeid2dofid(eid)
            groupeName2Dofxy[physName] = mesh.getEdgesCog()[eid]
        else: raise ValueError(funName+" not coded for space.interpolatoryAt.")
    return groupeName2DofId, groupeName2Dofxy

def setValues(vxy, evaluator):
    """ A simple function that return values associated to the points in array vxy

    according to the evaluator. The evaluator can be a scalar or a function
    """
    funName = setValues.__name__
    if type(evaluator) in [float, int, np.float64, np.int64]:
        values = np.ones(len(vxy))*evaluator
    elif callable(evaluator):
        values = evaluator(vxy)
    else:
        raise ValueError('evaluator of '+str(type(evaluator))+' unknown in '+funName+'.')
    return values

dirName2dirVal = {'x':0, 'y':1}
def computeDirichlet(space, physName2Dirichlet, groupeName2DofId, groupeName2Dofxy):
    ''' return the list of free dofs ids, fixed dofs ids and fixed dofs values '''
    funName = computeDirichlet.__name__
    fixedValues = np.array([])
    fixedDofs = np.array([], dtype=np.int64)
    for physName, dir2Value in physName2Dirichlet.items():
        dofid = groupeName2DofId[physName]
        dofxy = groupeName2Dofxy[physName]
        for directionName, value in dir2Value.items():
            direction = dirName2dirVal[directionName]
            if  direction is None:
                raise ValueError('directionName '+directionName+' unknown in '+funName+'.')
            fixedDofs = np.append(fixedDofs, dofid[:, direction])
            fixedValues = np.append(fixedValues, setValues(dofxy, value))
    nbfixed = len(fixedDofs)
    fixedDofs, index = np.unique(fixedDofs, return_index=True)
    #if nbfixed != len(fixedDofs):
    #    warn(' Some dofs where fixed Twice in '+funName+'.')
    nbfixed = len(fixedDofs)
    fixedValues = fixedValues[index]
    freeDofs = np.setdiff1d(np.arange(space.size()), fixedDofs)
    return freeDofs, fixedDofs, fixedValues

def computeNeumann(space, physName2Neumann:dict):
    ''' return the vector F such as F.T.dot(U) = int_neumann ( t.u_h) dGamma_h '''
    F = np.zeros((space.scalar_size(), space.vecdim))
    edgequad = quad.Edge_gauss(0)
    for physName, dir2Value in physName2Neumann.items():
        eids, ___ = space.m.getEdgeGroupe(physName)
        N = space.operator_dof2val_edge(eids, edgequad.s)
        xy = space.m.getEdgesCog(eids)
        l = space.m.getEdgesLength(eids)
        t = np.zeros((eids.size, 2))
        for directionName, value in dir2Value.items():
            direction = dirName2dirVal[directionName]
            if  direction is None:
                funName = computeNeumann.__name__
                raise ValueError('directionName '+directionName+' unknown in '+funName)
            t[:, direction] = l*setValues(xy, value)
        F += (edgequad.w*N.T.dot(t.flatten())).reshape((-1, 2))
    return F.flatten()

def computeVolumeLoad(space, source):
    """ compute the vector F such as F.U = int_Omega f.u_h dOmega_h """
    assert isinstance(source, np.ndarray)
    quadSource = quad.T3_gauss(0)
    N = space.operator_dof2val_tri(quadSource.uv)
    J = space.tris_map.J
    wSource = quadSource.w[0]*source[np.newaxis, :]*J[:, np.newaxis]
    Fint = N.T.dot(wSource.flatten())
    return Fint

class ElasticitySolver():
    """ E class to set up and compute a 2d linear elasticity problem with FEM method """
    def __init__(self, mesh, C=planeStrainC2222CSR(12., 7.),
                 physName2Dirichlet:None|dict=None,
                 physName2Neumann:None|dict=None,
                 source=np.array([0., 0.]),
                 spaceConstructor=fs.FEMSpaceP1, stab=0.):
        self.quadBilinForm = quad.T3_gauss(0)
        self.quadAtCorners = quad.T3_nodes()
        self.C = C
        self.stab = stab
        self.mesh = mesh
        self.setModel(spaceConstructor)
        self.physName2Dirichlet:dict = {} if physName2Dirichlet is None else physName2Dirichlet
        self.physName2Neumann:dict = {} if physName2Neumann is None else physName2Neumann
        self.groupeName2DofId, self.groupeName2Dofxy = getDofsAndInterpolationPoints(mesh,
                                                  self.physName2Dirichlet.keys(), self.space)
        self.freeDofs, self.fixedDofs, self.fixedValues = computeDirichlet(self.space,
                        self.physName2Dirichlet, self.groupeName2DofId, self.groupeName2Dofxy)
        self.Fneumann = computeNeumann(self.space, self.physName2Neumann)
        self.Fint = computeVolumeLoad(self.space, source)
        nbfixed = len(self.fixedDofs)
        if nbfixed < 3:
            warn(f' Only {nbfixed:d} fixed dofs: not enough to prevent rigid-body motion in {computeDirichlet.__name__}')
    def setC(self, C):
        self.C = C
        self.setD()
    def setD(self):
        J = self.space.tris_map.J
        nf = len(J)
        C = self.C.todense().flatten()
        wD = np.einsum("i, ij -> ij", self.quadBilinForm.w[0]*J,
                       C[np.newaxis, :]).reshape((-1, 4, 4))
        self.wD = st.regular_block_diag_to_csr(wD)
        D = np.broadcast_to(C, (nf, 16)).reshape((-1, 4, 4))
        self.D = st.regular_block_diag_to_csr(D)
    def setModel(self, spaceConstructor=fs.FEMSpaceP1):
        self.space = spaceConstructor("DisplacementSpace", self.mesh, vecdim=2)
        self.B = self.space.operator_dof2grad_tri(self.quadBilinForm.uv)
        self.NatVertices = self.space.operator_dof2val_tri(self.quadAtCorners.uv)
        self.setD()
    def setStabilisation(self, s, quadEdge):
        if not isinstance(self.space, fs.FEMSpaceP1NC):
            raise ValueError("Stabilisation term only implemented for P1NC")
        space = self.space
        lE = space.m.getEdgesLength()
        sTri2edge = space.operator_trival2edge().dot(0.5*space.tris_map.J).reshape((-1, 2))
        hE = np.sum(sTri2edge, axis=1)/2/lE
        N = space.operator_edgetrace(quadEdge)
        self.jumpOp = N[1::2] - N[:-1:2]
        weights = s*((lE/hE)[:, np.newaxis] * quadEdge.w).reshape(-1)
        DjumpBlock = (weights[:, np.newaxis]*np.eye(2).flatten()[np.newaxis, :]).reshape(-1, 2, 2)
        self.Djump = st.regular_block_diag_to_csr(DjumpBlock)

    def assemble(self):
        Fneumann = self.Fneumann
        Bfree = self.B[:, self.freeDofs]
        Bfixed = self.B[:, self.fixedDofs]
        U = np.zeros(self.space.size())
        U[self.fixedDofs] = self.fixedValues
        gradUdiri = Bfixed.dot(self.fixedValues)
        Fdiri = -Bfree.T.dot(self.wD.dot(gradUdiri))
        F = Fdiri + Fneumann[self.freeDofs] + self.Fint[self.freeDofs]
        Kfree = Bfree.T.dot(self.wD).dot(Bfree)
        if isinstance(self.space, fs.FEMSpaceP1NC):
            quadEdge = quad.Edge_gauss(2)
            self.setStabilisation(self.stab, quadEdge)
            internalEdges = self.space.m.getInternalEdges()
            vecDim = self.space.vecdim
            nptPerEdge = quadEdge.s.size
            rowsPerEdge = vecDim*nptPerEdge
            internalEdgesDofs = (vecDim*internalEdges[:, np.newaxis]
                                 +np.arange(vecDim)[np.newaxis, : ]).flatten()
            stabDofs = np.concatenate([internalEdgesDofs, self.fixedDofs])
            stabRows = np.sort(((rowsPerEdge*(stabDofs//vecDim)
                                 +stabDofs%vecDim)[:, np.newaxis]
                                +nptPerEdge*np.arange(vecDim)[np.newaxis, :]).flatten())
            Nedges = self.jumpOp[stabRows, :]
            NedgesFree = Nedges[:, self.freeDofs]
            NedgesFixed = Nedges[:, self.fixedDofs]
            Djump = self.Djump[stabRows, :][:, stabRows]
            SDiri = -NedgesFree.T.dot(Djump.dot(NedgesFixed.dot(self.fixedValues)))
            SFree = NedgesFree.T.dot(Djump.dot(NedgesFree))
            Kfree = Kfree + SFree
            F = F + SDiri
        return Kfree, F, U
    def solve(self, solver="cholmod"):
        Kfree, F, U = self.assemble()
        if solver == "default":
            U[self.freeDofs] = sp.sparse.linalg.spsolve(Kfree, F)
        elif solver == "cholmod":
            chol = sksparse.cholmod.cholesky(Kfree)
#            if compute_rcond:
#                diag = chol.D()
#                rcond = (np.min(diag)/np.max(diag))**2
            U[self.freeDofs] = chol(F)
        return U
    def postProDisplacement(self, U):
        return self.mesh.getTris2VerticesCoord(), self.NatVertices.dot(U).reshape((-1, 3, 2))
    def postProStress(self, U):
        return self.mesh.getTris2VerticesCoord(), self.D.dot(self.B.dot(U)).reshape((-1, 1, 2, 2))
    def postProStrain(self, U):
        return self.mesh.getTris2VerticesCoord(), self.B.dot(U).reshape((-1, 1, 2, 2))
    def L2StressError(self, U, exactStressFun, quadrature=quad.T3_gauss(0)):
        uv  = quadrature.uv
        weight = quadrature.w
        npt = len(uv)
        B = self.space.operator_dof2grad_tri(uv)
        gradNum = B.dot(U).reshape((-1, npt, 2, 2))
        C = self.C.todense().reshape((1,1,2,2,2,2))
        stressNum = np.einsum('egijkl, egkl-> egij', C, gradNum)
        NP1 = fs.FEMSpaceP1.N(uv)
        trisQuadPoints = np.einsum('ijk, ikl-> ijl', NP1[np.newaxis, :, :], self.mesh.getTris2VerticesCoord())
        stressExact = exactStressFun(trisQuadPoints)
        stressError2 = np.sum((stressNum - stressExact)**2, axis = (2,3))
        J = self.space.tris_map.J
        wStressError2 = J[:, np.newaxis]*weight[np.newaxis, :]*stressError2
        return np.sqrt(np.sum(wStressError2))

class testElasticity(unittest.TestCase):
    def testPatch(self):
        """ - patch test on a squared load by Dirchlet and then Neumann """
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)
        L = 0.7
        E = 12.2
        nu = 0.3
        ux = 0.98
        lamb, mu = YoungPoisson2LambdaNu(E, nu)
        C = planeStrainC2222CSR(lamb, mu)
        phys2Dirichlet = {'left': {'x':0.}, 'bottom': {'y': 0}, 'right': {'x': ux}}
        modelName, physical_dict = mg.square(length=L, relh=0.3)
        mesh = sm.gmshModel2sMesh(modelName, phys2Dirichlet.keys())
        pb = ElasticitySolver(mesh, C, physName2Dirichlet=phys2Dirichlet,
                                spaceConstructor=fs.FEMSpaceP1)
        U = pb.solve()
        ___, stress = pb.postProStress(U)
        e11 = ux/L
        s11 = 4*mu*(lamb+mu)/(lamb+2*mu)*e11
        stressExact = np.array([[s11, 0.], [0.,0.]])
        error = np.linalg.norm(stress - stressExact[np.newaxis, np.newaxis] )
        self.assertTrue(error < 1.e-9)
        phys2Dirichlet = {'left': {'x':0.}, 'bottom': {'y': 0}}
        phys2Neumann = {'right': {'x':s11}}
        pb = ElasticitySolver(mesh, C, physName2Dirichlet=phys2Dirichlet, physName2Neumann=phys2Neumann,
                                spaceConstructor=fs.FEMSpaceP1)
        e22 = -lamb*e11/(lamb+2*mu)
        strainExact = np.array([[e11, 0.], [0., e22]])
        U = pb.solve()
        ___, strain = pb.postProStrain(U)
        error = np.linalg.norm(strain - strainExact[np.newaxis, np.newaxis] )
        self.assertTrue(error < 1.e-9)
        gmsh.finalize()

def main(*, element_size=0.05, profile=True):
    """ main Program and example : elasticity on L shape domain """
    E = 1.
    nu = 0.4999
    lamb, mu = YoungPoisson2LambdaNu(E, nu)
    meshgenerator = mg.l_shape
    #meshgenerator = mg.square
    
    phys2Dirichlet = {'dOmega_left': {'x':0., 'y':0.}}
    phys2Neumann = {'dOmega_right':{'x': 0.01}}
    #phys2Dirichlet = {'left': {'x':0., 'y':0.}, 'right':{'x': 0.01, 'y':0.}}
    #phys2Neumann = {}
    spaceConstructors = [fs.FEMSpaceP1, fs.FEMSpaceP1NC]
    solver = "cholmod" # "default", "AATcholmod"
    popUpGmsh = True
    Profile.doprofile = profile
    gmsh.initialize()
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    gmsh.option.setNumber('Mesh.SurfaceEdges', 1 if element_size > 0.1 else 0)
    modelName, physical_dict = meshgenerator(element_size)
    edgeGroupeNames = list(phys2Dirichlet.keys())+ list(phys2Neumann.keys())
    mesh = sm.gmshModel2sMesh(modelName, edgeGroupeNames)
    nv, ne, nt = mesh.getNbVertices(), mesh.getNbEdges(), mesh.getNbTris()
    print('nv {:d}, ne {:d}, nf {:d}'.format(nv, ne, nt))
    C = planeStrainC2222CSR(lamb, mu)
    for spaceConstructor in spaceConstructors:
        spaceName = spaceConstructor.__name__+'_'
        Profile.enable()
        tic = time.perf_counter()
        pb = ElasticitySolver(mesh, C, physName2Dirichlet=phys2Dirichlet,
                              physName2Neumann=phys2Neumann,
                              spaceConstructor=spaceConstructor,
                              stab=2*mu*0.5)
                              
        U = pb.solve()
        toc = time.perf_counter()
        print(f"assemble +solve: {toc - tic:0.4f} seconds")
        Profile.disable()
        gp.listPlotFieldTri(*pb.postProStress(U), P0=True, viewname=spaceName+'Stress')
        gp.listPlotFieldTri(*pb.postProStrain(U), P0=True, viewname=spaceName+'Strain')
        gp.listPlotFieldTri(*pb.postProDisplacement(U), P0=False,
                            viewname=spaceName+'Displacement', VectorType="Displacement",
                            DisplacementFactor=1.)
        
        stress = pb.postProStress(U)[1]
        neumannEdges, neumannEdgeFlip = mesh.getEdgeGroupe('dOmega_other')
        if neumannEdges.size:
            e2t_s, e2t = mesh._e2t_()
            stressNeumann = stress[e2t[e2t_s[neumannEdges]]]
            normals = mesh.getEdgesNormal(neumannEdges, neumannEdgeFlip)
            tractionNeumann = np.einsum("i...jk,i...k->i...j", stressNeumann, normals)
            gp.listPlotFieldLine(mesh.getEdges2VerticesCoord()[neumannEdges], tractionNeumann,
                                 P0=True, viewname=spaceName+'bcTraction', VectorType="Arrow")
            print('Max neumann norm ', np.max(np.linalg.norm(tractionNeumann.squeeze(), axis=1)))

    if popUpGmsh:
        gmsh.fltk.run()
    gmsh.finalize()
    Profile.print_stats()

if __name__ == '__main__':
    main(element_size=0.01)
