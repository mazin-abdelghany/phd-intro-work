# Table of Contents

1. [PhD project](#phd-project)
2. [PhD topics - Year 1](#phd-topics---year-1)
3. [Running Code in Docker](#running-code-in-docker)
   - [R environment](#r-environment)
   - [Python environment](#python-environment)

# PhD project

## Bayesian optimisation for adaptive randomisation in clinical trials
Supervisors: Drs. David S. Robertson and Paul D. W. Kirk

Bayesian optimisation is a powerful tool in machine learning that is used to find the optimal parameters of a function that is expensive to evaluate[^ref1]. It is particularly useful in scenarios where the objective function is noisy, expensive, or has unknown properties. This technique leverages probabilistic models to make informed decisions about where to sample next, balancing exploration and exploitation.

One promising application of Bayesian optimisation that has not yet been explored is in optimising patient allocation for clinical trials. Traditionally, in a clinical trial, patients are randomised to different treatment arms using a fixed randomisation scheme (e.g., with equal probabilities). However, this approach can be inefficient and lead to suboptimal patient benefit properties for those within the trial. An alternative approach is to use adaptive randomisation, which updates the randomisation probabilities using the accumulating response data from the trial and patient covariates[^ref2]. This method can increase the statistical power of the trial and improve patient outcomes.

So far, optimal adaptive randomisation schemes have only been derived for simple trial settings and objective functions[^ref3][^ref4]. I plan to use Bayesian optimisation as a tool to find more complex and realistic optimal allocation targets for adaptive randomisation in clinical trials. Using Bayesian optimisation will allow the characterisation of adaptive randomisation schemes that are more efficient, have better patient benefit properties, and are tailored to the specific characteristics of the patients in the trial.

Working at the interface between machine learning and clinical trial methodology, I plan to implement Bayesian optimisation algorithms and adaptive randomisation schemes, and validate them through simulation studies and real-world clinical trial data. A key part of the project will also be the development of software packages to help enable the proposed methodology to be used in practice.

[^ref1]: Frazier (2018). “A Tutorial on Bayesian Optimization”, arXiv preprint,  https://arxiv.org/abs/1807.02811.  
[^ref2]: Robertson et al. (2023). “Response-adaptive randomization in clinical trials: from myths to practical considerations”, Statistical Science, https://doi.org/10.1214%2F22-STS865.  
[^ref3]: Rosenberger et al. (2001). “Optimal Adaptive Designs for Binary Response Trials”, Biometrics, https://doi.org/10.1111/j.0006-341X.2001.00909.x.  
[^ref4]: Tymofyeyev et al. (2007). “Implementing Optimal Allocation in Sequential Binary Response Experiments”, Journal of the American Statistical Association, https://www.jstor.org/stable/27639834.  

# PhD topics - Year 1

This repository will serve as a central location for coding examples, presentations, and derivations that I will use to explore the application of Bayesian optimization to the development of efficient study designs.

# Running Code in Docker

Note: the below environments are development environments. Therefore, the Docker containers that are used do not include a so-called "lockfile" (e.g., renv.lock or uv.lock). After completion of the project, a lockfile can easily be created in order to ensure that the development environment is portable to another Docker container without issue.

## R environment

To run RMarkdown files:
1. Ensure that you have Docker installed for your operating system.
2. Git clone the repository to your directory of choice.
3. `cd` into the directory `phd-bayesopt-esd/z_docker_R` 
4. Run `docker compose -f rstudio.yaml up` 
     - Depending on your system, this command may take several minutes.
5. If the Docker container has been built before and the `rstudio.yaml` file has changed, instead run `docker compose -f rstudio.yaml --build`
     - Depending on your system, this command may take several minutes.
7. In your browser of choice, open `http://localhost:8787`.
8. Type in the username `rstudio` and use the password in the environment variable within the `.yaml` file.

In order to see the files within the source directory:
1. Within the R console, `setwd("/project")`.
2. In the Files panel, under "More", click "Go To Working Directory". Any files in the source directory should appear here. Any files created within this directory, will also be saved in the source directory.

In order to run some notebooks, my personal Python package `groupSequentialDesigns` needs to be installed. Files for this package are contained in the directory `groupSequentialDesigns`. To install, run:  
``devtools::load_all(path = "../groupSequentialDesigns/")``  
within the R environment.

Open the RMarkdown file, and run as normal.

## Python environment 

This Python environment contains `tensorflow`, `GPflow`, and `Trieste`. Gaussian process regression is implemented using `GPflow`, a library build on `tensorflow`. Bayesian optimization is run using `Trieste`.

There are two options with which notebooks can be run:  
    - JupyterLab and notebooks  
    - Marimo notebooks

These two notebook environments are in two containers built using the same Dockerfile. When the Docker compose command is run, two containers will spin up simultaneously allowing the choice between entering a Jupyter notebook or a Marimo notebook.

To run the containers:
1. Ensure that you have Docker installed for your operating system system.
2. Git clone the repository to your directory of choice.
3. `cd` into the directory `phd-bayesopt-esd/z_docker_bayesopt`
5. Run `docker compose -f gpflow-trieste.yaml up`
    - Depending on your system, this command may take several minutes.
5. If the Docker container has been built before and the `gpflow-trieste.yaml` file has changed, instead run `docker compose -f gpflow-trieste.yaml --build`
     - Depending on your system, this command may take several minutes.
6. In your browser of choice, copy and paste one of the URLs outputted into the terminal.
8. Files in the directory should appear in the JupyterLab environment or the Marimo home screen.

Notebooks can be run as normal.

In order to run some notebooks, my personal Python package `py_group_sequential_designs` needs to be installed. Files for this package are contained in the directory `pyGroupSequentialDesigns`. To install, the Docker container running your notebook of choice needs to be entered.

1. Run `docker exec -it bayes-opt-jupyter-nb-1 bash` OR `docker exec -it bayes-opt-marimo-nb-1 bash`
    - Container names are automatically generated by compose. If you have named containers differently, replace either `bayes-opt-jupyter-nb-1` or `bayes-opt-marimo-nb-1` with those names.
2. Run `uv pip install /tf/pyGroupSequentialDesigns`. 
3. Output should look like:
```
Built py_group_sequential_designs @ file:///tf/pyGroupSequentialDesigns
Prepared 1 package . . .
Installed 1 package . . .
+ py-group-sequential-designs==0.0.1
```
4. Imports in Jupyter or Marimo `from py_group_sequential_designs import . . . ` should then run without issue.
