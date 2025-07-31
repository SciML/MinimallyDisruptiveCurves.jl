# MinimallyDisruptiveCurves

This is a toolbox implementing the algorithm introduced in [1]. **Documentation, examples, and user guide are found [here](https://dhruva2.github.io/MinimallyDisruptiveCurves.docs/).**

The package is a model analysis tool. It finds functional relationships between model parameters that best preserve model behaviour.

  - You provide a differentiable cost function that maps parameters to 'how bad the model behaviour is'. You also provide a locally optimal set of parameters θ*.

  - The package will generate curves in parameter space, emanating from θ*. Each point on the curve corresponds to a set of model parameters. These curves are 'minimally disruptive' with respect to the cost function (i.e. model behaviour).
  - These curves can be used to better understand interdependencies between model parameters, as detailed in the documentation.

[1] Raman, Dhruva V., James Anderson, and Antonis Papachristodoulou. "Delineating parameter unidentifiabilities in complex models." Physical Review E 95.3 (2017): 032314.
