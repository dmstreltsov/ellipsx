<h1 align='center'>Ellipsx</h1>

Ellipsx is a [JAX](https://github.com/jax-ml/jax) package for ellipsometry data analysis.

Features include:
- null-ellipsometry data analysis with the Jones matrix formalism (cf. H. Tompkins, E. A. Irene, Handbook of Ellipsometry, William Andrew, 2005)

## Installation

For installation in conda environment with [Jupyter](https://jupyter.org/), clone the ellipsx repository:
```
git clone https://github.com/dmstreltsov/ellipsx.git
```

Create conda environment from yaml file:
```
cd ellipsx
conda env create --file environment.yml
```

Activate this environment and run Jupyter notebook:

```
conda activate ellipsx
jupyter notebook
```

Do not forget to select a proper kernel in the Jupyter if you installed a few ones, i.e. the one from created conda environment `conda env: ellipsx`.

## Minimal installation
For minimal installation:
```
pip install git+https://github.com/dmstreltsov/ellipsx.git
```

Requires Python 3.10+, Jax and [Optimistix]{https://github.com/patrick-kidger/optimistix} for nonlinear optimization with Jax.


