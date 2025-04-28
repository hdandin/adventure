'''
 # @ Author: hdandin
 # @ Created on: 2024-06-25 17:06:52
 # @ Modified time: 2024-07-24 17:27:01
 '''

import time
from pathlib import Path
import numpy as np
import gmsh
from scipy.integrate import quad
import mechanics_solver as ms
import function_spaces as fs
import oneLevelSimplexMesh as sm
import fracture_analysis_line as fa
import gmsh_post_pro_interface as gp
import crack_features_plot_tools as pt
from execTools import Profile
from integration_domain import Contour
import interpolation as ip
import line_integration as li
import williams

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
    
    def impose(self, fract:fa.FractureAnalysisLine, eid=None, output_gmsh=False, filename=""):
        """ Impose fields at integration points """
        def u(s):
            return fract.get_williams_displ(*ip.interp2curve(s, 
                                                             fract.contour.get_vertex_coords(eid)
                                                             ).T)
        def stress(s):
            return fract.get_williams_stress(*ip.interp2curve(s, 
                                                              fract.contour.get_vertex_coords(eid)
                                                              ).T)
        def grad_u(s):
            return fract.get_williams_grad_displ(*ip.interp2curve(s, 
                                                                  fract.contour.get_vertex_coords(
                                                                      eid)).T)
        def strain(s):
            return fract.get_williams_strain(*ip.interp2curve(s, 
                                                              fract.contour.get_vertex_coords(eid)
                                                              ).T)
        return u, stress, grad_u, strain

class StressInterpError:
    """ Stress interpolation error: class for imposed fields
    
    U(N) = U_W(N) and stress(G) = stress_W(G)
    with
    - N: node
    - G: mesh integration point
    - U_W, stress_W: obtained by evaluation of Williams EEF for given n,a_n
    """
    def __init__(self, E:float, nu:float, hyp2d:str, mesh:sm.sMesh, xy_g:np.ndarray):
        self.E, self.nu = E, nu
        self.hyp2d = hyp2d
        self.mesh = mesh
        self.xy_g = xy_g

    def impose(self, fract:fa.FractureAnalysisLine, output_gmsh=False, filename=""):
        """ Boundary conditions for stress interpolation error: nodal displacement """
        U = fract.get_williams_displ(*self.mesh.xy.T)
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
    
    def impose(self, fract:fa.FractureAnalysisLine, output_gmsh=False, filename=""):
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
        B = pb.space.operator_dof2grad_tri(pb.quadBilinForm.uv)
        grad_u = B.dot(U).reshape((-1,2,2))
        strain = 0.5*(grad_u.swapaxes(-1,-2) + grad_u)
        stress = pb.D.dot(pb.B.dot(U)).reshape((-1,2,2))

        if output_gmsh:
            u_fix = np.zeros_like(U)
            u_fix[pb.fixedDofs] = pb.fixedValues
            gmsh.view.write(gmsh_export_displ(self.mesh, pb.NatVertices, u_fix,
                                              "fediscr-BC displ for FE"), 
                                              f"{filename}_fediscr.pos", append=True)
        
        return U.reshape((-1,2)), stress, grad_u, strain

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

    def loop_over_meshes(self, fe_u_interp:'function', fe_stress_interp:'function',
                         integ_method:str, error_type:'function',
                         cvge_analysis:bool=True, ortho_analysis:bool=False):
        """ Perform convergence analysis, loop over mesh sizes
        
        *Args:
        - fe_u_interp, fe_stress_interp : interpolation method for displacement (resp. stress) from 
        node (resp. integ point) to edge cog
        - integ_method : for bueckner integral, exact (Gaussian quad) or rectangle method
        - error_type : integration (IntegError), stress interpolation (StrsInterpError), FE 
        discretisation (FEDiscrError)
        - cvge_analysis : perform convergence analysis
        - ortho_analysis : perform orthogonality analysis
        """

        self.outfile_cvge = "results/cvge_" + error_type.__name__ + "_Integ" + integ_method

        if cvge_analysis:
            print(f'\n----- Convergence Analysis: {error_type.__name__} -----')

            self.all_williams_coefs = np.full((len(self.a_n),len(self.element_lengths)), np.nan)
            self.from_j = np.full((2,len(self.element_lengths)), np.nan)

            with open(self.outfile_cvge+"_coefs.txt", "w", encoding="utf8") as f:
                f.write('\nWilliams coefs imposed:\n')
                f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                                for x in zip(self.n,self.a_n,self.b_n)))
                
        if ortho_analysis:
            print(f'\n----- Orthogonality Analysis: {error_type.__name__} -----')

            self.all_bueckner_matrices = np.full((len(self.n), 2*len(self.n),
                                                  len(self.element_lengths)), np.nan)
            
            self.outfile_ortho = "results/ortho_" + error_type.__name__ + "_Integ" + integ_method

            self.cond = np.full((len(self.element_lengths)), np.nan)
            
            # load Williams coefs from convergence analysis
            if not cvge_analysis:
                npzfile = np.load(self.outfile_cvge+".npz")
                self.all_williams_coefs = npzfile['a_computed']

            with open(self.outfile_ortho+"_vals.txt", "w", encoding="utf8") as f:
                f.write('\nWilliams coefs imposed:\n')
                f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                                for x in zip(self.n,self.a_n,self.b_n)))
        
        tic = time.perf_counter()
        for ct, h in enumerate(self.element_lengths):

            modelname = f"MT_{h}"
            filename = f"results/msh/{modelname}"

            print(f'\nModel        : {modelname}')
            print('Started at '+time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
            tac = time.perf_counter()

            if self.output_gmsh:
                gmsh_outfilename = f"{filename}_{error_type.__name__}.pos"
                Path(gmsh_outfilename).unlink(missing_ok=True)

            if not Path(f"msh/{modelname}.msh").is_file():
                print('\n   no mesh found')
            else:

                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal",0)

                mesh, xy_g, nat_vertices, co = self.get_mesh_and_contour(modelname)

                if cvge_analysis:
                    if self.output_gmsh:
                        self.compute_williams_coefs(ct, error_type, fe_u_interp, fe_stress_interp, 
                                                    integ_method, co, mesh, xy_g, 
                                                    nat_vertices, gmsh_outfilename)
                    else:
                        self.compute_williams_coefs(ct, error_type, fe_u_interp, fe_stress_interp, 
                                                    integ_method, co, mesh, xy_g)
                
                if ortho_analysis:
                    self.compute_bueckner_integ(ct, error_type, fe_u_interp, fe_stress_interp, 
                                                integ_method, co, mesh, xy_g)

                gmsh.finalize()

                toc = time.perf_counter()
                print(f'# current mesh analysis performed in {toc - tac:0.4f} seconds')
        
        print(f'\n# Convergence analysis performed in {toc - tic:0.4f} seconds')

    def compute_williams_coefs(self, ct, error_type, fe_u_interp, fe_stress_interp, integ_method, 
                               co, mesh, xy_g, nat_vertices=None, gmsh_outfilename=None):
        """ Compute Williams coefficient for current mesh size and save results """
        
        u_imp, stress_imp, fa_cvg, a_1_from_j, j = self.run_fracture_analysis(fe_u_interp, 
                                                                fe_stress_interp, 
                                                                integ_method, error_type, 
                                                                co, mesh, xy_g)

        if self.output_gmsh:
            gmsh.view.write(gmsh_export_displ(mesh, nat_vertices, u_imp.flatten(),
                                                error_type.__name__+'-Imposed displ'),
                                                gmsh_outfilename)
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
            if j is not None:
                f.write(f'\nFrom J-integral: A_1 = {a_1_from_j:.4f} ; J = {j:.4f}')

    def get_mesh_and_contour(self, modelname):
        """ Get mesh properties and create Contour """

        tic = time.perf_counter()

        gmsh.open(f"msh/{modelname}.msh")
        edge_groupe_names = [gmsh.model.getPhysicalName(dim, tag) 
                            for dim, tag in gmsh.model.getPhysicalGroups()]
        mesh = sm.gmshModel2sMesh(modelname, edge_groupe_names)

        pb = self.create_empty_solver(mesh)
        N = pb.space.operator_dof2val_tri(pb.quadBilinForm.uv)
        nat_vertices = pb.NatVertices

        # contour properties
        contour_edges, _ = mesh.getEdgeGroupe('contour')
        contour_vertices = mesh.getEdges2Vertices(contour_edges)
        trisuv = N.dot(mesh.xy.flatten()).reshape((-1,2))
        co = Contour(contour_edges, contour_vertices, mesh, trisuv)
        
        print(f'Contour info : {len(contour_edges):d} edges, \
              {len(contour_vertices):d} vertices')
        
        tac = time.perf_counter()
        print(f'    # mesh management performed in {tac - tic:0.4f} seconds')
        
        return mesh, trisuv, nat_vertices, co

    def run_fracture_analysis(self, fe_u_interp, fe_stress_interp, integ_method, error_type, co, 
                              mesh, xy_g):
        """ Run fracture analysis for given mesh and error type """
        
        tic = time.perf_counter()

        # imposed field (reference)
        fa_imp = fa.FractureAnalysisLine(co, self.E, self.nu, self.n, self.hyp2d)
        fa_imp.a_n = self.a_n
        fa_imp.b_n = self.b_n
        
        imposed_fields = error_type(self.E, self.nu, self.hyp2d, mesh, xy_g)

        if "integ" in error_type.__name__.lower() and integ_method.lower() == "exact":

            tac = time.perf_counter()
            print(f'    # imposed field performed in {tac - tic:0.4f} seconds')

            # fracture analysis
            fa_cvg = fa.FractureAnalysisLine(co, self.E, self.nu, self.n, self.hyp2d)

            for n in self.n:
                m = -n

                # Bueckner I
                edge_integ = np.zeros(co.get_nb_edges())
                for eid in range(co.get_nb_edges()): # eid = local index
                    u_imp, stress_imp,grad_u_imp, strain_imp = imposed_fields.impose(fa_imp, eid)
                    line_integ = li.LineIntegral(co, fa_imp.kappa, fa_imp.mu, u_imp, stress_imp, 
                                                 fe_u_interp, fe_stress_interp, integ_method, 
                                                 grad_u_imp, strain_imp)
                    edge_integ[eid] = line_integ.integrate_bueckner_exactly_on_edge(eid, m, 
                                                                                          1., 0.)
                bueckner_i = np.dot(co.get_lengths(), edge_integ)
                a_n = - fa_imp.mu/(fa_imp.kappa + 1) * 1/(np.pi*n*(-1)**(n+1)) * bueckner_i

                # Bueckner II
                edge_integ = np.zeros(co.get_nb_edges())
                for eid in range(co.get_nb_edges()): # eid = local index
                    u_imp, stress_imp, grad_u_imp, strain_imp = imposed_fields.impose(fa_imp, eid)
                    line_integ = li.LineIntegral(co, fa_imp.kappa, fa_imp.mu, u_imp, stress_imp, 
                                                    fe_u_interp, fe_stress_interp, integ_method,
                                                    grad_u_imp, strain_imp)
                    edge_integ[eid] = line_integ.integrate_bueckner_exactly_on_edge(eid, m, 
                                                                                          0., 1.)
                bueckner_ii = np.dot(co.get_lengths(), edge_integ)
                b_n = - fa_imp.mu/(fa_imp.kappa + 1) * 1/(np.pi*n*(-1)**(n+1)) * bueckner_ii

                fa_cvg.a_n.append(a_n)
                fa_cvg.b_n.append(b_n)

            # J-integral
            edge_integ = np.zeros(co.get_nb_edges())
            for eid in range(co.get_nb_edges()): # eid = local index
                u_imp, stress_imp,grad_u_imp, strain_imp = imposed_fields.impose(fa_imp, eid)
                line_integ = li.LineIntegral(co, fa_imp.kappa, fa_imp.mu, u_imp, stress_imp, 
                                             fe_u_interp, fe_stress_interp, integ_method, 
                                             grad_u_imp, strain_imp)
                edge_integ[eid] = line_integ.integrate_j_exactly_on_edge(eid)
            j  = np.dot(co.get_lengths(), edge_integ)
            k_1 = (j*self.E)**0.5
            a_1_from_j = k_1 / (2*np.pi)**0.5

            print('j',j)
            print('a1_from_j',a_1_from_j)

            print(f'    # fracture analysis performed in {time.perf_counter() - tac:0.4f} seconds')
        else:
            u_imp, stress_imp, grad_u_imp, stress_u_imp = imposed_fields.impose(fa_imp)

            tac = time.perf_counter()
            print(f'    # imposed field performed in {tac - tic:0.4f} seconds')

            # fracture analysis
            fa_cvg = fa.FractureAnalysisLine(co, self.E, self.nu, self.n, self.hyp2d)
            a_1_from_j, j = fa_cvg.run(u_imp, stress_imp, fe_u_interp, fe_stress_interp, integ_method, 
                                    grad_u_imp, stress_u_imp)
            
            print(f'    # fracture analysis performed in {time.perf_counter() - tac:0.4f} seconds')

        return u_imp, stress_imp, fa_cvg, a_1_from_j, j

    def create_empty_solver(self, mesh):
        """ Create empty mechanical solver to get QoI """
        phys2dirichlet = {}
        phys2neumann = {}
        lamb, mu = fa.lame_constants(self.E, self.nu, self.hyp2d)
        C = ms.planeStrainC2222CSR(lamb, mu)
        pb = ms.ElasticitySolver(mesh, C, physName2Dirichlet=phys2dirichlet,
                                 physName2Neumann=phys2neumann,
                                 spaceConstructor=fs.FEMSpaceP1)
        return pb
    
    def print_mesh_specs(self):
        """ Print mesh specifications: element length, number of elements for LI """

        for h in self.element_lengths:
            modelname = f"MT_{h}"

            print(f'\nModel        : {modelname}')
            print('Started at '+time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))

            if not Path(f"msh/{modelname}.msh").is_file():
                print('\n   no mesh found')
            else:
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal",0)
                self.get_mesh_and_contour(modelname)
                gmsh.finalize()

    def compute_bueckner_integ(self, ct, error_type, fe_u_interp, fe_stress_interp, integ_method, 
                               co, mesh, xy_g):
        """ Compute Bueckner LI for current mesh size and save results """
        
        self.all_bueckner_matrices, self.all_kronecker_matrices = self.run_orthogonality_analysis(self.all_williams_coefs[:,ct], 
                                                                     error_type,
                                                                     fe_u_interp, fe_stress_interp, integ_method, co, mesh, xy_g)

        # save result
        np.savez(self.outfile_ortho+".npz", element_lengths=self.element_lengths, orders=self.n,
                 B=self.all_bueckner_matrices, delta=self.all_kronecker_matrices, cond=self.cond)
        
        with open(self.outfile_ortho+"_vals.txt", "a", encoding="utf8") as f:
            f.write('\nElement size: '+str(self.element_lengths[ct]))
            f.write('\nBueckner matrix computed:\n')
            for line in self.all_bueckner_matrices:
                line2=','.join(["{0:.4f}".format(x) for x in line])
                f.write(line2+'\n')
        
    def run_orthogonality_analysis(self, a_n, error_type, fe_u_interp, fe_stress_interp, 
                                   integ_method, co, mesh, xy_g):
        """ Analyse the loss of orthogonality due to numerical approximation on Bueckner-Chen integral evaluation """
        
        order_max = np.max(self.n)
        orders_to_be_computed = np.concatenate((np.arange(-order_max,0,dtype=float),
                                                np.arange(1,order_max+1,dtype=float)))
        nb_orders = 2*order_max

        kappa = fa.kolosov_constant(self.nu, self.hyp2d)
        _, mu = fa.lame_constants(self.E, self.nu, self.hyp2d)
        imposed_fields = error_type(self.E, self.nu, self.hyp2d, mesh, xy_g)

        fract = fa.FractureAnalysisLine(co, self.E, self.nu, [np.inf], self.hyp2d)
        fract.a_n = np.append(a_n[::-1], a_n)
        fract.b_n = [0.]*nb_orders
        a_m = 1.
        b_m = 0.

        bueckner = np.zeros((nb_orders, nb_orders))
        delta = np.zeros_like(bueckner)
        for i,n in enumerate(orders_to_be_computed):
            fract.williams_orders = [n]
            for j,m in enumerate(orders_to_be_computed):
                if "integ" in error_type.__name__.lower() and integ_method.lower() == "exact":
                    edge_integ = np.zeros(co.get_nb_edges())
                    for eid in range(co.get_nb_edges()): # eid = local index
                        U, stress, _, _ = imposed_fields.impose(fract, eid)
                        line_integ = li.LineIntegral(co, kappa, mu, 
                                                     U, stress,
                                                     ip.UInterpolatorNone,
                                                     ip.StressInterpolatorNone,  "exact", _, _)
                        edge_integ[eid] = line_integ.integrate_bueckner_exactly_on_edge(eid, m, a_m, b_m)
                        
                    bueckner[i,j] = np.dot(co.get_lengths(), edge_integ)
                    delta[i,j] = - bueckner[i,j] * mu / (np.pi*(kappa+1) * (-1)**(n+1) * n * fract.a_n[i])
                    if n != 0 and m == -n:
                            print(f'n={n:.0f}, m={m:.0f}, delta={delta[i,j]:.3e}')
                            print(f'    imp - comp = {self.a_n[i%len(self.n)]-fract.a_n[i]:.4f}')
                    else:
                        print(f'n={n:.0f}, m={m:.0f}, B_nm={bueckner[i,j]:.3e}')
                else:
                    U, stress, _, _ = imposed_fields.impose(fract)
                    line_integ = li.LineIntegral(co, fract.kappa, fract.mu, U, stress, 
                                                 fe_u_interp, fe_stress_interp, integ_method, _, _)
                    bueckner[i,j] = line_integ.solve_bueckner(m, a_m, b_m)
                    delta[i,j] = - bueckner[i,j] * mu / (np.pi*(kappa+1) * (-1)**(n+1) * n * fract.a_n[i])

        return bueckner, delta
    
    @classmethod
    def compute_circle_exact_integration(cls):
        """ Exact integration for a contour with parameter dtheta """

        r = 1.
        theta = -np.pi, np.pi

        outfile = "results/cvge_circle_ExactInteg"
        with open(outfile+"_coefs.txt", "w", encoding="utf8") as f:
            f.write('\nWilliams coefs imposed:\n')
            f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                            for x in zip(cls.n,cls.a_n,cls.b_n)))

        _, mu = fa.lame_constants(cls.E, cls.nu, cls.hyp2d)
        kappa = fa.kolosov_constant(cls.nu, cls.hyp2d)

        def imposed_stress(n, a_n, b_n, r, theta):
            sig = np.zeros((2,2))
            for i, a_i, b_i in zip(n, a_n, b_n):
                sig += williams.get_stress(i, a_i, b_i, r, theta).reshape((2,2))
            return sig
        def imposed_displ(n, a_n, b_n, r, theta, kappa, mu):
            u = np.zeros(2)
            for i, a_i, b_i in zip(n, a_n, b_n):
                u += williams.get_displ(i, a_i, b_i, r, theta, kappa, mu).reshape(2)
            return u
        def bueckner_chen(theta, r, kappa, mu, n, a_n, b_n, m, a_m, b_m):
            """ Brueckner-Chen integrand, theta is a float """
            sig_1 = imposed_stress(n, a_n, b_n, r, theta)
            u_1 = imposed_displ(n, a_n, b_n, r, theta, kappa, mu)
            sig_2 = williams.get_stress(m, a_m, b_m, r, theta).reshape((2,2))
            u_2 = williams.get_displ(m, a_m, b_m, r, theta, kappa, mu).reshape(2)
            normal = np.array([np.cos(theta), np.sin(theta)])
            bueckner = (sig_1.dot(u_2) - sig_2.dot(u_1)).dot(normal)
            return bueckner
        
        result = np.zeros((len(cls.a_n),1))
        A = np.zeros_like(cls.a_n)
        B = np.zeros_like(cls.b_n)
        for i, n in enumerate(cls.n):
            bueckner_i, _ = quad(bueckner_chen, theta[0], theta[1], 
                                 args=(r, kappa, mu, cls.n, cls.a_n, cls.b_n, -n, 1., 0.))
            
            A[i] = - mu/(kappa + 1) * 1/(np.pi*n*(-1)**(n+1)) * bueckner_i
            result[i] = A[i]
            print('n =',n)
            print('rel error on a_n:', abs(cls.a_n[i] - A[i])/cls.a_n[i])

            bueckner_ii, _ = quad(bueckner_chen, theta[0], theta[1], 
                                  args=(r, kappa, mu, cls.n, cls.a_n, cls.b_n, -n, 0., 1.))
            
            B[i] = - mu/(kappa + 1) * 1/(np.pi*n*(-1)**(n+1)) * bueckner_ii
            print('abs error on b_n:', abs(cls.b_n[i] - B[i]))

        # save results
        np.savez(outfile+".npz", element_lengths=[0], orders=cls.n, a_known=cls.a_n, a_computed=result)
        
        with open(outfile+"_coefs.txt", "a", encoding="utf8") as f:
            f.write('\nWilliams coefs computed:\n')
            f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                                for x in zip(cls.n, A, B)))
            f.write('\n')
            f.write('\n'.join(f'n = {x[0]:d}, |A_imp-A_comp| = {abs(x[1]-x[2]):.6e}'
                                for x in zip(cls.n, cls.a_n, A)))

if __name__ == "__main__":

    lengths = [1.6, 0.8, 0.4, 0.2, 0.1]

    Profile.doprofile = False

    Profile.enable()
    cvg = Convergence(lengths, output_gmsh=False)

    ## mesh specs
    cvg.print_mesh_specs()

    ## integration error
    cvg.compute_circle_exact_integration()
    cvg.loop_over_meshes(ip.UInterpolatorNone, ip.StressInterpolatorNone, "Exact", IntegError)
    pt.plot_cvge_relerror_log("IntegError_IntegExact")
    cvg.loop_over_meshes(ip.UInterpolatorNone, ip.StressInterpolatorNone, "Exact", IntegError,
                         cvge_analysis=False, ortho_analysis=True)

    # cvg.loop_over_meshes(ip.UInterpolatorNone, ip.StressInterpolatorNone, "Rect", IntegError)
    # pt.plot_cvge_relerror_log("IntegError_IntegRect")
    # cvg.loop_over_meshes(ip.UInterpolatorNone, ip.StressInterpolatorNone, "Rect", IntegError,
    #                      cvge_analysis=False, ortho_analysis=True)

    
    ## stress interpolation error
    # cvg.loop_over_meshes(ip.UInterpolatorLinear, ip.StressInterpolatorNeighboursMean, 
    #                      "Exact", StressInterpError)
    # cvg.loop_over_meshes(ip.UInterpolatorLinear, ip.StressInterpolatorLinearND, 
    #                      "Exact", StressInterpError)
    cvg.loop_over_meshes(ip.UInterpolatorLinear, ip.StressInterpolatorLinearND, 
                         "Rect", StressInterpError)
    pt.plot_cvge_relerror_log("StressInterpError_IntegRect")


    ## FE interpolation error
    # cvg.loop_over_meshes(ip.UInterpolatorLinear, ip.StressInterpolatorLinearND, 
    #                      "Exact", FEDiscrError)
    cvg.loop_over_meshes(ip.UInterpolatorLinear, ip.StressInterpolatorLinearND, 
                         "Rect", FEDiscrError)
    pt.plot_cvge_relerror_log("FEDiscrError_IntegRect")

    Profile.disable()

    Profile.print_stats()
