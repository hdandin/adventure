'''
 # @ Author: hdandin
 # @ Created on: 2024-06-25 17:06:52
 # @ Modified time: 2024-07-24 17:27:01
 '''

import time
from pathlib import Path
import numpy as np
import gmsh

import quadrature as quad
from mapping import MappingT3
import function_spaces as fs
import oneLevelSimplexMesh as sm
import mechanics_solver as ms
import gmsh_post_pro_interface as gp
from execTools import Profile

from integration_domain import Domain
import fracture_analysis as fa
import crack_features_plot_tools as pt
from volume_integration import EquivalentDomainIntegral

a = 50. # crack length
R = 10. # contour radius

def gmsh_export_displ(mesh, nat_vertices, U, viewname):
    """ Create gmsh view for .pos export of displacement """
    coord = mesh.getTris2VerticesCoord()
    u = nat_vertices.dot(U).reshape((-1, 3, 2))
    gv = gp.listPlotFieldTri(coord, u, P0=False, viewname=viewname,
                                VectorType="Displacement", DisplacementFactor=1.)
    return gv

def gmsh_export_stress(mesh, stress, viewname):
    """ Create gmsh view for .pos export of stress """
    gv = gp.listPlotFieldTri(mesh.getTris2VerticesCoord(), stress.reshape((-1, 1, 2, 2)),
                                P0=True, viewname=viewname)
    return gv

class IntegError:
    """ Integration error: class for imposed fields

    U(G) = U_W(G) and stress(G) = stress_W(G)
    with 
    - G: edge integration point (barycentre or Gauss point, depending on integration method)
    - U_W, stress_W: obtained by evaluation of Williams EEF for given n,a_n
    """
    def __init__(self, E:float, nu:float, hyp2d:str, mesh:sm.sMesh, xy_g:np.ndarray):
        self.E, self.nu = E, nu
        self.hyp2d = hyp2d
        self.mesh = mesh
        self.xy_g = xy_g
    
    def impose(self, fract:fa.FractureAnalysis, output_gmsh=False, filename=""):
        """ Impose fields at integration points """
        U = fract.get_williams_displ(*self.xy_g.T)
        stress = fract.get_williams_stress(*self.xy_g.T)
        
        # QoI for J-integral (only for n=1)
        grad_u = fract.get_williams_grad_displ(*self.xy_g.T)
        strain = 0.5*(grad_u.swapaxes(-1,-2) + grad_u)

        return U, stress, grad_u, strain

class FEDiscrError:
    """ FE discretisation error: class for imposed fields
    
    U(N) and stress(G)
    with
    - N: node
    - G: integration point
    - U_W, stress_W: obtained by FE simulation with Dirichlet boundary conditions U_imp
    - U_imp: evaluation of Williams EEF for given n,a_n on all free surfaces of the geometry
    """
    def __init__(self, E:float, nu:float, hyp2d:str, mesh:sm.sMesh, xy_g:np.ndarray):
        self.E, self.nu = E, nu
        self.hyp2d = hyp2d
        self.mesh = mesh
        self.xy_g = xy_g
    
    def impose(self, fract:fa.FractureAnalysis, output_gmsh=False, filename=""):
        """ Perform FE simulation with imposed displacement on boundary nodes

        1- generates Williams displacement field -> u_williams
        2- takes nodal displacements on boundary nodes -> fixed_dofs, fixed_values
        3- solve mechanical problem with boundary conditions computed in step 2
        """
        # generate Williams displacement field
        u_williams = fract.get_williams_displ(*self.mesh.xy.T).flatten()

        # initialise mechanics solver
        pb = self.create_empty_solver()

        # get dofs and nodal displacements on boundaries
        edge_groupe_names = ['left','top','right','bottom']
        pb.groupeName2DofId, pb.groupeName2Dofxy = ms.getDofsAndInterpolationPoints(
            self.mesh, edge_groupe_names, pb.space)
        fixed_values = np.array([])
        fixed_dofs = np.array([], dtype=np.int64)
        for phys_name in edge_groupe_names:
            dofid = pb.groupeName2DofId[phys_name].flatten()
            fixed_dofs = np.append(fixed_dofs, dofid)
            fixed_values = np.append(fixed_values, u_williams[dofid])
        fixed_dofs, index = np.unique(fixed_dofs, return_index=True)

        # apply Dirichlet boundary conditions
        pb.freeDofs = np.setdiff1d(np.arange(pb.space.size()), fixed_dofs)
        pb.fixedDofs = fixed_dofs
        pb.fixedValues = fixed_values[index]

        # solve problem
        U = pb.solve()

        # compute displacements, gradients and stresses at integration points
        N = pb.space.operator_dof2val_tri(pb.quadBilinForm.uv)
        B = pb.space.operator_dof2grad_tri(pb.quadBilinForm.uv)
        Ug = N.dot(U).reshape((-1,2))
        grad_u = B.dot(U).reshape((-1,2,2))
        strain = 0.5*(grad_u.swapaxes(-1,-2) + grad_u)
        stress = pb.D.dot(pb.B.dot(U)).reshape((-1,2,2))

        if output_gmsh:
            u_fix = np.zeros_like(U)
            u_fix[pb.fixedDofs] = pb.fixedValues
            gmsh.view.write(gmsh_export_displ(self.mesh, pb.NatVertices, u_fix,
                                              "fediscr-BC displ for FE"), 
                                              f"{filename}_fediscr.pos", append=True)
        
        return Ug, stress, grad_u, strain

    def create_empty_solver(self):
        """ Create empty mechanical solver to get QoI """
        phys2dirichlet = {}
        phys2neumann = {}
        lamb, mu = fa.lame_constants(self.E, self.nu, self.hyp2d)
        C = ms.planeStrainC2222CSR(lamb, mu)
        pb = ms.ElasticitySolver(self.mesh, C, physName2Dirichlet=phys2dirichlet,
                                 physName2Neumann=phys2neumann,
                                 spaceConstructor=fs.FEMSpaceP1)
        return pb


class Convergence:
    """ Class for convergence study of fracture analysis """

    hyp2d = "plane stress"
    E = 1.
    nu = 0.3
    n = [1, 2, 3, 4]
    a_n = [R**(1/2-i/2) for i in range(1,len(n)+1)]
    b_n = [0., 0., 0., 0.]

    def __init__(self, element_lengths:list, output_gmsh:bool=False):
        self.element_lengths = element_lengths
        self.output_gmsh = output_gmsh

    def loop_over_meshes(self, error_type:'function', mesh_type:str='', ng:int=0,
                         cvge_analysis:bool=True, ortho_analysis:bool=False, resfolder:str=''):
        """ Perform convergence analysis, loop over mesh sizes
        
        *Args:
        - error_type : integration (IntegError), stress interpolation (StrsInterpError), FE 
        discretisation (FEDiscrError)
        - mesh_type : for EDI (nicemesh, Xtreme)
        - ng : order of Gaussian quadrature
        - cvge_analysis : perform convergence analysis
        - ortho_analysis : perform orthogonality analysis
        """

        self.outfile_cvge = f"results/cvge_{error_type.__name__}_order{ng}"
        if cvge_analysis:
            print(f'\n----- Convergence Analysis: {error_type.__name__} -----')

            self.all_williams_coefs = np.full((len(self.n), len(self.element_lengths)), np.nan)
            self.from_j = np.full((2,len(self.element_lengths)), np.nan)

            with open(self.outfile_cvge+"_coefs.txt", "w", encoding="utf8") as f:
                f.write('\nWilliams coefs imposed:\n')
                f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                                for x in zip(self.n,self.a_n,self.b_n)))
                
        if ortho_analysis:
            print(f'\n----- Orthogonality Analysis: {error_type.__name__} -----')

            self.all_bueckner_matrices = np.full((len(self.n), 2*len(self.n),
                                                  len(self.element_lengths)), np.nan)
            
            self.outfile_ortho = f"results/ortho_{error_type.__name__}_order{ng}"

            self.cond = np.full((len(self.element_lengths)), np.nan)
            
            # load Williams coefs from convergence analysis
            if not cvge_analysis:
                self.outfile_cvge = f"results/{resfolder}/cvge_{error_type.__name__}_order{ng}"
                npzfile = np.load(self.outfile_cvge+".npz")
                self.all_williams_coefs = npzfile['a_computed']

            with open(self.outfile_ortho+"_vals.txt", "w", encoding="utf8") as f:
                f.write('\nWilliams coefs imposed:\n')
                f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                                for x in zip(self.n,self.a_n,self.b_n)))
        
        tic = time.perf_counter()
        for ct, h in enumerate(self.element_lengths):

            if mesh_type:
                modelname = f"MT_edi_{mesh_type}_{h}"
            else:
                modelname = f"MT_edi_{h}"
            filename = f"results/msh/{modelname}"

            print(f'\nModel        : {modelname}')
            print('Started at '+time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
            tac = time.perf_counter()

            if self.output_gmsh:
                gmsh_outfilename = f"{filename}_{error_type}.pos"
                Path(gmsh_outfilename).unlink(missing_ok=True)

            if not Path(f"msh/{modelname}.msh").is_file():
                print('\n   no mesh found')
            else:

                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal",0)

                mesh, xy_g, nat_vertices, dom = self.get_mesh_and_contour(modelname)

                if cvge_analysis:
                    if self.output_gmsh:
                        self.compute_williams_coefs(ct, error_type, dom, mesh, xy_g, 
                                                    nat_vertices, gmsh_outfilename)
                    else:
                        self.compute_williams_coefs(ct, error_type, dom, mesh, xy_g)
                
                if ortho_analysis:
                    self.compute_bueckner_edi(ct, error_type, dom, mesh, xy_g)

                gmsh.finalize()

                toc = time.perf_counter()
                print(f'# current mesh analysis performed in {toc - tac:0.4f} seconds')
        
        print(f'\n# Convergence analysis performed in {toc - tic:0.4f} seconds')

    def compute_williams_coefs(self, ct, error_type, dom, mesh, xy_g, 
                               nat_vertices=None, gmsh_outfilename=None):
        """ Compute Williams coefficient for current mesh size and save results """

        u_imp, stress_imp, fa_cvg, a_1_from_j, j = self.run_fracture_analysis(error_type, 
                                                                                dom, mesh, 
                                                                                xy_g)

        if self.output_gmsh:
            gmsh.view.write(gmsh_export_displ(mesh, nat_vertices, u_imp.flatten(),
                                                error_type.__name__+'-Imposed displ'),
                                                gmsh_outfilename, append=True)
            if error_type.__name__ != 'IntegError':
                gmsh.view.write(gmsh_export_stress(mesh, stress_imp,
                                                    error_type.__name__+'-Imposed stress'),
                                                    gmsh_outfilename, append=True)
            u = fa_cvg.get_williams_displ(*mesh.xy.T)
            stress = fa_cvg.get_williams_stress(*xy_g.T)
            gmsh.view.write(gmsh_export_displ(mesh, nat_vertices, u.flatten(),
                                                error_type.__name__+'-Williams displ'),
                                                gmsh_outfilename, append=True)
            gmsh.view.write(gmsh_export_stress(mesh, stress,
                                                error_type.__name__+'-Williams stress'),
                                                gmsh_outfilename, append=True)
            
        # save result
        for n in range(len(self.n)):
            self.all_williams_coefs[n,ct] = fa_cvg.a_n[n]
            self.from_j[0,ct] = a_1_from_j
            self.from_j[1,ct] = j

        np.savez(self.outfile_cvge+".npz", element_lengths=self.element_lengths, orders=self.n,
                    a_known=self.a_n, a_computed=self.all_williams_coefs, a_1_from_j=self.from_j)
            
        with open(self.outfile_cvge+"_coefs.txt", "a", encoding="utf8") as f:
            f.write('\nElement size: '+str(self.element_lengths[ct]))
            f.write('\nWilliams coefs computed:\n')
            f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.4f}, B_n = {x[2]:.4f}'
                                for x in zip(self.n, fa_cvg.a_n, fa_cvg.b_n)))
            f.write(f'\nFrom J-integral: A_1 = {a_1_from_j} ; J = {j}')

    def get_mesh_and_contour(self, modelname, gauss_order=0):
        """ Get mesh properties and create Contour """

        tic = time.perf_counter()

        gmsh.open(f"msh/{modelname}.msh")
        edge_groupe_names = [gmsh.model.getPhysicalName(dim, tag) 
                            for dim, tag in gmsh.model.getPhysicalGroups()]
        mesh = sm.gmshModel2sMesh(modelname, edge_groupe_names)

        quad_form = quad.T3_gauss(gauss_order) # sets Gaussian quadrature order to 0
        w_g = quad_form.w

        tris_map = MappingT3(mesh.getTris2VerticesCoord())
        space = fs.FEMSpaceP1('EDIspace', mesh, 2, tris_map)
        N = space.operator_dof2val_tri(quad_form.uv)
        xy_g = N.dot(mesh.xy.flatten()).reshape((-1,2))
        nat_vertices = space.operator_dof2val_tri(quad.T3_nodes().uv)

        dom = Domain(mesh, xy_g, tris_map.J, w_g)
        
        print(f'Domain info : {dom.get_nb_tris():d} triangles, \
              {len(dom.integpts):d} integration points')
        
        tac = time.perf_counter()
        print(f'    # mesh management performed in {tac - tic:0.4f} seconds')
        
        return mesh, xy_g, nat_vertices, dom

    def run_fracture_analysis(self, error_type, dom, mesh, trisuv):
        """ Run fracture analysis for given mesh and error type """
        
        tic = time.perf_counter()

        # imposed field (reference)
        fa_imp = fa.FractureAnalysis(dom, self.E, self.nu, self.n, self.hyp2d)
        fa_imp.a_n = self.a_n
        fa_imp.b_n = self.b_n
        
        imposed_fields = error_type(self.E, self.nu, self.hyp2d, mesh, trisuv)

        u_imp, stress_imp, grad_u_imp, strain_imp = imposed_fields.impose(fa_imp)

        tac = time.perf_counter()
        print(f'    # imposed field performed in {tac - tic:0.4f} seconds')

        # fracture analysis
        fa_cvg = fa.FractureAnalysis(dom, self.E, self.nu, self.n, self.hyp2d)
        a_1_from_j, j = fa_cvg.run(u_imp, stress_imp, grad_u_imp, strain_imp)
        
        print(f'    # fracture analysis performed in {time.perf_counter() - tac:0.4f} seconds')

        return u_imp, stress_imp, fa_cvg, a_1_from_j, j
    
    def print_mesh_specs(self, mesh_type:str=''):
        """ Print mesh specifications: element length, number of elements for EDI """

        for h in self.element_lengths:
            if mesh_type:
                modelname = f"MT_edi_{mesh_type}_{h}"
            else:
                modelname = f"MT_edi_{h}"

            print(f'\nModel        : {modelname}')
            print('Started at '+time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))

            if not Path(f"msh/{modelname}.msh").is_file():
                print('\n   no mesh found')
            else:
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal",0)
                self.get_mesh_and_contour(modelname)
                gmsh.finalize()

    def compute_bueckner_edi(self, ct, error_type, dom, mesh, xy_g):
        """ Compute Bueckner EDI for current mesh size and save results """
        
        self.all_bueckner_matrices, self.all_kronecker_matrices = self.run_orthogonality_analysis(self.all_williams_coefs[:,ct], 
                                                                     error_type, dom, mesh, xy_g)

        # save result
        np.savez(self.outfile_ortho+".npz", element_lengths=self.element_lengths, orders=self.n,
                 B=self.all_bueckner_matrices, delta=self.all_kronecker_matrices, cond=self.cond)
        
        with open(self.outfile_ortho+"_vals.txt", "a", encoding="utf8") as f:
            f.write('\nElement size: '+str(self.element_lengths[ct]))
            f.write('\nBueckner matrix computed:\n')
            for line in self.all_bueckner_matrices:
                line2=','.join(["{0:.4f}".format(x) for x in line])
                f.write(line2+'\n')
        
    def run_orthogonality_analysis(self, a_n, error_type, dom, mesh, xy_g):
        """ Analyse the loss of orthogonality due to numerical approximation on Bueckner-Chen integral evaluation """

        order_max = np.max(self.n)
        orders_to_be_computed = np.concatenate((np.arange(-order_max,0,dtype=float),
                                                np.arange(1,order_max+1,dtype=float)))
        nb_orders = 2*order_max

        kappa = fa.kolosov_constant(self.nu, self.hyp2d)
        _, mu = fa.lame_constants(self.E, self.nu, self.hyp2d)
        imposed_fields = error_type(self.E, self.nu, self.hyp2d, mesh, xy_g)

        fract = fa.FractureAnalysis(dom, self.E, self.nu, [np.inf], self.hyp2d)
        fract.a_n = np.append(a_n[::-1], a_n)
        fract.b_n = [0.]*nb_orders
        a_m = 1.
        b_m = 0.

        bueckner = np.zeros((nb_orders, nb_orders))
        delta = np.zeros_like(bueckner)
        for i,n in enumerate(orders_to_be_computed):
            fract.williams_orders = [n]
            for j,m in enumerate(orders_to_be_computed):
                U, stress, _, _ = imposed_fields.impose(fract)
                edi = EquivalentDomainIntegral(dom, kappa, mu, U, stress)
                bueckner[i,j] = edi.solve_bueckner(m, a_m, b_m)
                delta[i,j] = - bueckner[i,j] * mu / (np.pi*(kappa+1) * (-1)**(n+1) * n * fract.a_n[i])

        return bueckner, delta

if __name__ == "__main__":

    lengths = [1.6, 0.8, 0.4, 0.2, 0.1]

    Profile.doprofile = False

    Profile.enable()
    cvg = Convergence(lengths)

    ## mesh specs
    cvg.print_mesh_specs(mesh_type='Xtreme10')

    ## integration error
    cvg.loop_over_meshes(IntegError)
    pt.plot_cvge_relerror_log("IntegError_order0")
    cvg.loop_over_meshes(IntegError, cvge_analysis=False, ortho_analysis=True)

    ## FE interpolation error
    cvg.loop_over_meshes(FEDiscrError, mesh_type='Xtreme10')
    pt.plot_cvge_relerror_log("FEDiscrError_order0")
    cvg.loop_over_meshes(FEDiscrError, cvge_analysis=False, ortho_analysis=True)

    Profile.disable()

    Profile.print_stats()
