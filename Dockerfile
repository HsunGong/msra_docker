ARG CUDA="11.0"  
# will change to 10.2 when philly upgrades drivers
ARG CUDNN="8"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

# apt list -a xxx
# curl -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN (apt-get update || echo "Warning") \
    && apt-get install --no-install-recommends -y wget curl \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-get update

ENV CUDNN_VERSION=8.0.4.30-1+cuda11.0
ENV NCCL_VERSION=2.8.3-1+cuda11.0
RUN apt-get install --no-install-recommends -y  --allow-change-held-packages apt-utils git ca-certificates bzip2 cmake tree htop bmon iotop g++ \
    && apt-get install --no-install-recommends -y  --allow-change-held-packages libglib2.0-0 libsm6 libxext6 libxrender-dev sox \
    && apt-get install --no-install-recommends -y  --allow-change-held-packages openmpi-bin jq openssh-server infiniband-diags libibverbs-dev vim \
    && apt-get install --no-install-recommends -y  --allow-change-held-packages libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION} libcudnn8=${CUDNN_VERSION} libcudnn8-dev=${CUDNN_VERSION} \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda and set up python
ENV PATH=/miniconda/bin:$PATH
#ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
#ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh 1>/dev/null \
    && chmod +x /miniconda.sh \
    && /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh \
    && /miniconda/bin/conda install -y conda-build mamba python=3.8 \
    && /miniconda/bin/conda clean -ya

# Install Open MPI
ENV OPENMPI_VERSIONBASE=4.1
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.0
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/openmpi-${OPENMPI_VERSION}.tar.gz 1>/dev/null && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

ARG CUDA
ENV CUDA_HOME=/usr/local/cuda
RUN conda activate torch \
    && conda install -c pytorch -y scipy pytorch torchvision torchaudio cudatoolkit=${CUDA} \
    && conda clean -ya \
    && pip install pyyaml editdistance tensorboard_logger tensorboard pandas pymongo tensorflow \
    && HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod \
    && pip install py3nvml sentencepiece unidecode soundfile librosa \
    && pip install https://github.com/kpu/kenlm/archive/master.zip \
    && pip install numba==0.48 \
    && git clone https://github.com/SeanNaren/warp-ctc && cd warp-ctc && mkdir build && cd build && cmake .. && make && cd ../pytorch_binding && python setup.py install \
    && pip install fast_rnnt

# export FT_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DFT_WITH_CUDA=OFF"
# export FT_MAKE_ARGS="-j"
RUN cd /code \
    && git clone https://github.com/danpovey/fast_rnnt.git \
    && cd fast_rnnt \
    && conda activate torch \
    && python setup.py install \
    && python3 -c "import fast_rnnt; print(fast_rnnt.__version__)"

