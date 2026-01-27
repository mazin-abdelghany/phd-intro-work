# syntax=docker/dockerfile:1

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Set the locale

RUN apt-get update && apt-get upgrade -y
RUN apt-get -y install locales

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8  
ENV LANGUAGE=en_US:en  
ENV LC_ALL=en_US.UTF-8     

# Install necessary R apt packages

RUN apt-get install -y --no-install-recommends \
    bash-completion \
    ca-certificates \
    file \
    fonts-texgyre \
    g++ \
    gfortran \
    gsfonts \
    libblas-dev \
    libbz2-* \
    libcurl4 \
    "libicu[0-9][0-9]" \
    liblapack-dev \
    libpcre2* \
    libjpeg-turbo* \
    libpangocairo-* \
    libpng16* \
    libreadline8 \
    libtiff* \
    liblzma* \
    libxt6 \
    make \
    tzdata \
    unzip \
    zip \
    zlib1g \
    curl \
    default-jdk \
    devscripts \
    libbz2-dev \
    libcairo2-dev \
    libcurl4-openssl-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libicu-dev \
    libpcre2-dev \
    libpng-dev \
    libreadline-dev \
    libtiff5-dev \
    liblzma-dev \
    libx11-dev \
    libxt-dev \
    perl \
    rsync \
    subversion \
    tcl-dev \
    tk-dev \
    texinfo \
    texlive-extra-utils \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-recommended \
    texlive-latex-extra \
    x11proto-core-dev \
    xauth \
    xfonts-base \
    xvfb \
    wget \
    zlib1g-dev

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Update and install some necessary TF packages

RUN apt-get update
RUN apt-get install -y gnupg software-properties-common

# Add TF sources

# Install Nvidia repo keys
# See: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

# Add ppa:deadsnakes/ppa for better Python support on older Ubuntu releases
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update

# Add CPU requirements

RUN apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    software-properties-common

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*


# Set up python and install tensorflow

ARG PYTHON_VERSION=python3.11
ARG TENSORFLOW_PACKAGE=tf-nightly
COPY setup.packages.sh /setup.packages.sh
COPY setup.python.sh /setup.python.sh
RUN /setup.python.sh ${PYTHON_VERSION} 
RUN pip install --no-cache-dir ${TENSORFLOW_PACKAGE} 

# Install and setup jupyter
COPY jupyter.requirements.txt /jupyter.requirements.txt
RUN python3 -m pip install --no-cache-dir -r /jupyter.requirements.txt -U
RUN jupyter server extension enable --py jupyter_http_over_ws
RUN python3 -m ipykernel.kernelspec

# Install R
RUN apt-get update
RUN apt-get install r-base-dev -y \
	&& apt-get clean \
	&& apt-get remove \
	&& rm -rf /var/lib/apt/lists/*

# Set default R CRAN repo
RUN echo 'options("repos"="http://cran.rstudio.com")' >> /usr/lib/R/etc/Rprofile.site

# Install R packages and kernel for Jupyter notebook
RUN Rscript -e "install.packages(c('ggplot2', 'gplite', 'mvtnorm', 'bench', 'rpact', 'profvis', 'tictoc'))"
RUN Rscript -e "install.packages('plotly')"
RUN Rscript -e "install.packages('IRkernel')"

# to register the kernel in the current R installation
RUN Rscript -e "IRkernel::installspec()"

# install Python packages of interest
# rpy2 allows interfacing between R and Python
RUN pip install rpy2 pandas plotly
RUN pip install gpflow
RUN pip install trieste trieste[plotting]

# Run the notebook

WORKDIR /tf
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]