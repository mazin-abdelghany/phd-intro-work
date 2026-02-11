# syntax=docker/dockerfile:1

# pull the Ubuntu Noble base image
FROM docker.io/library/ubuntu:jammy

# set the language (this can be changed as needed)
ENV LANG=en_US.UTF-8

# set the R version that is desired
# set the Python version that is desired
ENV R_VERSION=4.5.2
ENV PYTHON_VERSION=3.11

# update the base container image
RUN apt-get update && apt-get upgrade -y

# install required system dependencies
# curl to download R from posit.co
# dpkg to obtain system architecture
# locales to set LC_* in R
# ca-certificates to confirm downloads
RUN apt-get install --no-install-recommends curl \
  dpkg \
  locales \
  ca-certificates -y

# download the R version specified above
# for the correct CPU architecture (using dpkg)
RUN curl -O https://cdn.posit.co/r/ubuntu-2204/pkgs/r-${R_VERSION}_1_$(dpkg --print-architecture).deb

# install R
# this automatically installs system dependencies
RUN apt-get install --no-install-recommends ./r-${R_VERSION}_1_$(dpkg --print-architecture).deb -y

# update the base container image
RUN apt-get update && apt-get upgrade -y

# set the locale
RUN /usr/sbin/locale-gen --lang "${LANG}"
RUN /usr/sbin/update-locale --reset LANG="${LANG}"

# Clean up
RUN rm -rf /var/lib/apt/lists/*

# add R path to the environment PATH
# installation using .deb file addes bin to below directory
ENV PATH="$PATH:/opt/R/4.5.2/bin"

# set default R CRAN repo
RUN echo 'options("repos"="https://cloud.r-project.org")' >> opt/R/4.5.2/lib/R/etc/Rprofile.site

# install R packages and kernel for Jupyter notebook
RUN Rscript -e "install.packages(c('ggplot2', 'gplite', 'mvtnorm', 'bench'))"
RUN Rscript -e "install.packages(c('rpact', 'profvis', 'tictoc', 'plotly', 'IRkernel'))"

# update and install some necessary packages for tensorflow
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget gnupg software-properties-common

# add TF sources
# install Nvidia repo keys
# see: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

# add CPU requirements
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    software-properties-common

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /bin/uv

# ensure that uv installs into usr/bin
ENV UV_PYTHON_BIN_DIR=/usr/local/bin

# create a uv virtual environment 
# with the specified python version
# at the directory /opt/venv
RUN uv venv --python ${PYTHON_VERSION} /opt/venv

# use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/venv

# place entry points in the environment at the front of the path
ENV PATH="/opt/venv/bin:$PATH"

# install required packages for Bayesian optimization
# these will ve added to the venv created above
# setuptools is pinned to 80.10.2 because gpflow needs
# pkg_resources, which was removed from setuptools 81.0.0 and on
# IPython also must be pinned to version 8 or less for
# gpflow print_summary utility error
RUN uv pip install pandas numpy matplotlib \
  jupyter jupyter_http_over_ws \
  tensorflow gpflow 'setuptools==80.10.2' \
  ipython==8.38.0 \
  trieste trieste[plotting] marimo \
  rpy2 plotly

# setup jupyter as server
RUN jupyter server extension enable --py jupyter_http_over_ws

# to register the kernel for Python and R in Jupyter
RUN python3 -m ipykernel.kernelspec
RUN Rscript -e "IRkernel::installspec()"

WORKDIR /tf

# expose the ports for two notebooks
EXPOSE 8888
EXPOSE 8080

# run commands for notebooks are in docker compose YAML