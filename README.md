# Adventure

This repositery contains the codes produced within the ANR Adventure project for the extraction of Williams higher-order coefficients from the Bueckner-Chen integral, as well as the corresponding convergence analysis [1].

The study aims to:
- implement Bueckner-Chen as a line integral,
- write and implement a new formulation as an Equivalent Domain Integral (EDI),
- perform a convergence analysis and evaluate the ability of the method to provides Williams higher-order coefficients (and comparison with the $J$-integral).

The Finite Element solver used for the computations is PY_XFEM (developped by Nicolas Chevaugeon, available at  [https://gitlab.com/c4506/py_xfem](https://gitlab.com/c4506/py_xfem)). It is not necessary to clone PY_XFEM to use the Adventure code: this repository contains all the necessary files.

[1] Héloïse Dandin, Nicolas Chevaugeon, Julien Réthoré. Convergence analysis of crack features extraction using conjugate work integral. 2025. [⟨hal-05018756⟩)](https://hal.science/hal-05018756v1)

### Dependencies
numpy, scipy, gmsh

### Usage

The mesh file is created with Gmsh in `mesh_crack.py`: it includes physical tags for the integration domain.

Main files for crack features' extraction:
- extract Williams coefficients for a given problem: `crack_features_line.py` and `crack_features_edi.py`
- run convergence analysis using given mesh files: `crack_features_line_convergence.py` and `crack_features_edi_convergence.py`

Files for crack features extraction:
- `fracture_analysis.py`: extract Williams coefficients for given FE field
- `volume_integration.py`: compute Bueckner-Chen EDI
- `williams.py`: Williams eigenfunctions for displacement and stress
- `integration_domain.py` : tools for handling integration domain
- `interpolation.py`: interpolation displacement and stress fields from and to different locations
- `crack_features_plot_tools.py`: plot convergence results
- `fracture_analysis_line.py`, `line_integration.py`: equivalent for line integration
- `test_*.py`: test files (with unittest)

### Convergence analysis

Williams fields (with known coefficients, to be retrieved) are applied at various locations, leading to different types of errors (see [1] for more details):
- integration error: FE fields are known at integration points
- error due to stress interpolation (for line integral): stress fields are known at 2D elements' integration points
- error due to FE discretisation: Williams fields are known on the specimen outer boundaries and a linear elastic simulation is performed to get FE fields