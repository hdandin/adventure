'''
 # @ Author: hdandin
 # @ Created on: 2024-05-31 10:05:54
 # @ Modified time: 2024-09-06 15:10:57

 '''

import gmsh
import mesh_crack as mc
import mechanics_solver as ms
import function_spaces as fs
import oneLevelSimplexMesh as sm
import gmsh_post_pro_interface as gp
from integration_domain import Domain
import fracture_analysis as fa
from execTools import Profile

# uimp = 2.
fimp = 10e3
width, height, a, r1, r2, h1 = 100., 200., 50., 10., 10.8, 0.8
h2 = 10*h1
meshgenerator = mc.mesh_with_crack_edi
E, nu = 72000., 0.33
hyp2d = "plane stress"

gmsh.initialize()
gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
gmsh.option.setNumber("General.Terminal",0)
gmsh.option.setNumber('Mesh.SurfaceEdges', 1 if h1 > 0.1 else 0)

### mesh
modelname, physdict = meshgenerator(width*2, height, a, r1, r2, h1, h2, translate=(-a,0.), 
                                    modelname='MT_edi_'+str(h1))
# BCs enforce symmetry, no rigid body motion
phys2dirichlet = {'crack_plane':{'y': 0.}, 'crack_tip':{'x': 0.}}
phys2neumann = {'bottom':{'y': -fimp},'top':{'y': fimp}}

filename = f"results/2d_problem/{modelname}"

groupe_names = physdict.keys()
mesh = sm.gmshModel2sMesh(modelname, groupe_names)
print(f'Info   : {modelname} {mesh.getNbVertices():d} nodes, {mesh.getNbEdges():d} edges, \
        {mesh.getNbTris():d} elements')

SpaceConstructor = fs.FEMSpaceP1
lamb, mu = fa.lame_constants(E, nu, hyp2d)
C = ms.planeStrainC2222CSR(lamb, mu)

pb = ms.ElasticitySolver(mesh, C, physName2Dirichlet=phys2dirichlet,
                        physName2Neumann=phys2neumann,
                        spaceConstructor=SpaceConstructor)
U = pb.solve()

gvstress = gp.listPlotFieldTri(*pb.postProStress(U), P0=True, viewname='Stress')
gvstrain = gp.listPlotFieldTri(*pb.postProStrain(U), P0=True, viewname='Strain')
gvdispl = gp.listPlotFieldTri(*pb.postProDisplacement(U), P0=False, viewname='Displacement', 
                              VectorType="Displacement", DisplacementFactor=1.)

gmsh.write(filename+".msh")
gmsh.view.write(gvdispl, filename+".pos")
gmsh.view.write(gvstrain, filename+".pos", append=True)
gmsh.view.write(gvstress, filename+".pos", append=True)

### Fracture analysis

Profile.doprofile = False

stress = pb.D.dot(pb.B.dot(U)).reshape((-1,2,2))
N = pb.space.operator_dof2val_tri(pb.quadBilinForm.uv)
xy_g = N.dot(mesh.xy.flatten()).reshape((-1,2))
J = pb.space.tris_map.J
wg = pb.quadBilinForm.w
Ug = N.dot(U).reshape((-1,2))
B = pb.space.operator_dof2grad_tri(pb.quadBilinForm.uv)
grad_u = B.dot(U).reshape((-1,2,2))
strain = 0.5*(grad_u.swapaxes(-1,-2) + grad_u)
Profile.enable()

dom = Domain(mesh, xy_g, J, wg)
fract = fa.FractureAnalysis(dom, E, nu, [1,2,3,4,5], hyp2d)
a_1_from_j, _ = fract.run(Ug, stress, grad_u, strain)
Profile.disable()

print('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                for x in zip(fract.williams_orders, fract.a_n, fract.b_n)))
print(f'\nFrom J-integral: A_1 = {a_1_from_j}')

u_w = fract.get_williams_displ(*mesh.xy.T)
gv_u_w = gp.listPlotFieldTri(*pb.postProDisplacement(u_w.reshape(-1)), P0=False, 
                                viewname='Displacement Williams', VectorType="Displacement", 
                                DisplacementFactor=1.)

stress_w = fract.get_williams_stress(*xy_g.T)
gv_stress_w = gp.listPlotFieldTri(mesh.getTris2VerticesCoord(), stress_w, P0=True, 
                                    viewname='Stress Williams')
gmsh.view.write(gv_u_w, filename+".pos", append=True)
gmsh.view.write(gv_stress_w, filename+".pos", append=True)

gmsh.finalize()

with open(filename+'.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(f'n = {x[0]:d}, A_n = {x[1]:.2f}, B_n = {x[2]:.2f}'
                        for x in zip(fract.williams_orders, fract.a_n, fract.b_n)))

Profile.print_stats()
