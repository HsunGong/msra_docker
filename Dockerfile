FROM pytorch/pytorch:1.11.0-cuda11.0-cudnn8-devel

### apt support
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev sox wget \
 && apt-get install -y -q openmpi-bin jq openssh-server \
 && apt-get install -y -q infiniband-diags \
 && apt-get install -y -q libibverbs-dev \
 && apt-get install -y -q vim

### CUDA SUPPORT
ENV CUDA_HOME=/usr/local/cuda
ENV CUDNN_VERSION=8.0.4.30-1+cuda11.0
ENV NCCL_VERSION=2.8.3-1+cuda11.0
RUN apt install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION} \
    && apt install -y libcudnn8=${CUDNN_VERSION} libcudnn8-dev=${CUDNN_VERSION}

### python support
RUN pip install scikit-learn pyyaml editdistance tensorboard_logger tensorboard pandas pymongo
RUN pip install py3nvml sentencepiece unidecode soundfile librosa
RUN pip install https://github.com/kpu/kenlm/archive/master.zip
RUN pip install numba==0.48
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod
RUN git clone https://github.com/SeanNaren/warp-ctc && cd warp-ctc && mkdir build && cd build && cmake .. && make && cd ../pytorch_binding && python setup.py install

# Install Open MPI
ENV OPENMPI_VERSIONBASE=4.1
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.4
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi