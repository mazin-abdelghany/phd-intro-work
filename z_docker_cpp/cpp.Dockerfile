# syntax=docker/dockerfile:1

# pull Ubuntu 22.04
FROM docker.io/library/ubuntu:jammy

# update the base container image
RUN apt-get update && apt-get upgrade -y

# Install the compiler and build tools
RUN apt-get install --no-install-recommends -y \
    build-essential \
    ninja-build \
    gcc \
    g++ \
    make \
    cmake \
    clang \
    lldb \
    gdb \
    git \
    curl \
    wget \
    unzip \
    pkg-config \
    valgrind \
    ca-certificates \
    software-properties-common \git \
    curl \
    wget \
    unzip \
    pkg-config \
    valgrind \
    ca-certificates \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project