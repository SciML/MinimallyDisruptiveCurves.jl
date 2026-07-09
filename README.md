# MinimallyDisruptiveCurves

[![Tests](https://github.com/SciML/MinimallyDisruptiveCurves.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/SciML/MinimallyDisruptiveCurves.jl/actions/workflows/Tests.yml)
[![Documentation](https://github.com/SciML/MinimallyDisruptiveCurves.jl/actions/workflows/Documentation.yml/badge.svg)](https://SciML.github.io/MinimallyDisruptiveCurves.jl)
[![Codecov](https://codecov.io/gh/SciML/MinimallyDisruptiveCurves.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/MinimallyDisruptiveCurves.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blue)](https://github.com/SciML/ColPrac)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgevals/by_name/M/MinimallyDisruptiveCurves.svg)](https://juliaci.github.io/NanosoldierReports/pkgevals/by_name/M/MinimallyDisruptiveCurves/)

This is a toolbox implementing the algorithm introduced in [1]. **Documentation, examples, and user guide are found [here](https://dhruva2.github.io/MinimallyDisruptiveCurves.docs/).**

A Python version using Diffrax and Jax is [here](https://pypi.org/project/minimally-disruptive-curves/). It's less tested and necessarily uses a slightly worse curve evolution algorithm due to more limited callback functionalities in the Python ODE solvers.

The package is a model analysis tool. It finds functional relationships between model parameters that best preserve model behaviour.

  - You provide a differentiable cost function that maps parameters to 'how bad the model behaviour is'. You also provide a locally optimal set of parameters θ*.

  - The package will generate curves in parameter space, emanating from θ*. Each point on the curve corresponds to a set of model parameters. These curves are 'minimally disruptive' with respect to the cost function (i.e. model behaviour).
  - These curves can be used to better understand interdependencies between model parameters, as detailed in the documentation.

[1] Raman, Dhruva V., James Anderson, and Antonis Papachristodoulou. "Delineating parameter unidentifiabilities in complex models." Physical Review E 95.3 (2017): 032314.
