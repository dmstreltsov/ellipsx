<h1 align='center'>Ellipsx</h1>

Ellipsx is a [JAX](https://github.com/jax-ml/jax) package for ellipsometry data analysis.

Features include:
- null-ellipsometry data analysis with the Jones matrix formalism (cf. H. Tompkins, E. A. Irene, Handbook of Ellipsometry, William Andrew, 2005)


## Installation

For installation in a conda environment for usage with [Jupyter](https://jupyter.org/), clone the ellipsx repository:
```
git clone https://github.com/dmstreltsov/ellipsx.git
```

Create conda environment from `environment.yml' file in this repository:
```
cd ellipsx
conda env create --file environment.yml
```

Activate this conda environment and install Jupyter kernel:
```
conda activate ellipsx
ipython kernel install --user --name=ellipsx-kernel
```

Run Jupyter notebook (system installed or from another conda environment) and select `ellipsx-kernel`.





