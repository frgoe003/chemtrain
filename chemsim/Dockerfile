# Use the TensorFlow build image as the base
FROM tensorflow/build:2.17-python3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libfftw3-dev \
    libjpeg-dev \
    libpng-dev \
    libhdf5-dev \
    libopenmpi-dev \
    openmpi-bin \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-h5py \
    python3-yaml \
    python3-lxml \
    python3-six \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual python environment
ENV PYENV=/venv
RUN python -m venv $PYENV
ENV PATH="$PYENV/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install "jax[cuda12]==0.4.30" \

