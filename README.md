# Table of Contents

1. [PhD project](#phd-project)
2. [PhD topics - Year 1](#phd-topics---year-1)
3. [Running Code in Docker](#running-code-in-docker)
   - [R environment](#r-environment)
   - [Python environment with `TensorFlow`, `GPflow`, and `Trieste`](#python-environment-with-tensorflow-gpflow-and-trieste)

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

## R environment

To run RMarkdown files:
1. Ensure that you have Docker installed for your system.
2. Git clone the repository to your directory of choice.
3. `cd` to the directory in which you have cloned the repository.
4. Run `docker build -f [insert-name].Dockerfile -t [insert-tag] .` (ensure that the line ends in a `.`).  
     - Depending on your system, this command may take several minutes.
6. Run the command `docker run --rm --mount type=bind,source=.,dst=/project -ti -p 8787:8787 [insert tag]`.
7. In your browser of choice, open `http://localhost:8787`.
8. Type in the username `rstudio` and use the auto-generated password from the command line.

In order to see the files within the source directory:
1. Within the R console, `setwd("/project")`.
2. In the Files panel, under "More", click "Go To Working Directory". Any files in the source directory should appear here. Any files created within this directory, will also be saved in the source directory.

Open the RMarkdown file, and run as normal.

## Python environment with `TensorFlow`, `GPflow`, and `Trieste`

In Week 5 and onwards, Gaussian process (GP) regression is implemented using `GPflow`, a library built in Python on TensorFlow. Bayesian optimization is run using `Trieste`.

To run Jupyter notebooks:
1. Ensure that you have Docker installed for your system.
2. Git cloe the repository to your directory of choice.
3. `cd` to the director in which you have cloned the repository.
4. `cd` into `z_docker_setup`.
5. Run `docker compose -f gpflow.yaml up`.
6. In your browser of choice, open `http://localhost:8889`.
7. Copy and paste the token outputted in the terminal after the `docker compose` command executed above
8. Files in the directory should appear in the JupyterLab environment.

The folders with format `YYYY-MM_tX` should be available within the Jupyter environment. Notebooks can be run as normal.
