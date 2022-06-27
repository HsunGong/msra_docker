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
    && apt-get update 2>&1 1>/dev/null

ENV CUDNN_VERSION=8.0.4.30-1+cuda11.0
ENV NCCL_VERSION=2.8.3-1+cuda11.0
RUN apt-get install --no-install-recommends -y  --allow-change-held-packages apt-utils git ca-certificates bzip2 cmake tree htop bmon iotop g++ 2>&1 1>/dev/null \
    && apt-get install --no-install-recommends -y  --allow-change-held-packages libglib2.0-0 libsm6 libxext6 libxrender-dev sox 2>&1 1>/dev/null \
    && apt-get install --no-install-recommends -y  --allow-change-held-packages openmpi-bin jq openssh-server infiniband-diags libibverbs-dev vim 2>&1 1>/dev/null \
    && apt-get install --no-install-recommends -y  --allow-change-held-packages libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION} libcudnn8=${CUDNN_VERSION} libcudnn8-dev=${CUDNN_VERSION} 2>&1 1>/dev/null \
    && apt-get autoclean && rm -rf /var/lib/apt/lists/*

# Install miniconda and set up python
ENV PATH=/miniconda/bin:$PATH
#ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
#ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN wget --no-verbose -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && chmod +x /miniconda.sh \
    && /miniconda.sh -b -p /miniconda 2>&1 1>/dev/null \
    && rm /miniconda.sh \
    && /miniconda/bin/conda install -y conda-build python=3.8 \
    && /miniconda/bin/conda clean -ya

# Install Open MPI
ENV OPENMPI_VERSIONBASE=4.1
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.0
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget --no-verbose -q https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default 2>&1 1>/dev/null && \
    make -j $(nproc) all 2>&1 1>/dev/null && \
    make install 2>&1 1>/dev/null && \
    ldconfig && \
    rm -rf /tmp/openmpi

ARG CUDA
ENV CUDA_HOME=/usr/local/cuda
RUN conda install pytorch torchvision torchaudio cudatoolkit=${CUDA} -c pytorch \
    && conda clean -ya \
    && pip --no-cache-dir install scipy pyyaml editdistance tensorboard_logger tensorboard pandas pymongo \
    && pip --no-cache-dir install py3nvml sentencepiece unidecode soundfile librosa \
    && pip install https://github.com/kpu/kenlm/archive/master.zip \
    && pip --no-cache-dir install numba==0.48

RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
# RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod
# && git clone https://github.com/SeanNaren/warp-ctc && cd warp-ctc && mkdir build && cd build && cmake .. 2>&1 1>/dev/null && make 2>&1 1>/dev/null && cd ../pytorch_binding && python setup.py install \

# export FT_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DFT_WITH_CUDA=OFF"
# export FT_MAKE_ARGS="-j"
RUN cd /code \
    && git clone https://github.com/danpovey/fast_rnnt.git \
    && cd fast_rnnt \
    && python setup.py install \
    && python3 -c "import fast_rnnt; print(fast_rnnt.__version__)"
