ARG CUDA="11.0"  
# will change to 10.2 when philly upgrades drivers
ARG CUDNN="8"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev sox wget

RUN apt-get update -y \
 && apt-get install -y -q openmpi-bin jq openssh-server \
 && apt-get install -y -q infiniband-diags \
 && apt-get install -y -q libibverbs-dev \
 && apt-get install -y -q vim

# apt list -a xxx
ENV CUDNN_VERSION=8.0.4.30-1+cuda11.0
ENV NCCL_VERSION=2.8.3-1+cuda11.0
RUN apt-key del 7fa2af80 \
    && curl -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm -f cuda-keyring_1.0-1_all.deb \
    && apt update \
    && apt install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION} \
    && apt install -y libcudnn8=${CUDNN_VERSION} libcudnn8-dev=${CUDNN_VERSION}



# Install miniconda and set up python
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name TorchSpeech python=3.8 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=TorchSpeech
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install Open MPI
ENV OPENMPI_VERSIONBASE=4.1
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.0
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

ARG CUDA
RUN conda install -c pytorch -y scipy pytorch torchvision torchaudio cudatoolkit=${CUDA} \
 && conda clean -ya
RUN pip install pyyaml editdistance tensorboard_logger tensorboard pandas pymongo tensorflow
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod
RUN pip install py3nvml sentencepiece unidecode soundfile librosa
RUN pip install https://github.com/kpu/kenlm/archive/master.zip
RUN pip install numba==0.48
ENV CUDA_HOME=/usr/local/cuda
RUN git clone https://github.com/SeanNaren/warp-ctc && cd warp-ctc && mkdir build && cd build && cmake .. && make && cd ../pytorch_binding && python setup.py install


#RUN cp warp-ctc/pytorch_binding/setup.py /
#RUN rm -rf warp-ctc
#RUN git clone https://github.com/jnishi/warp-ctc && cd warp-ctc && mkdir build && cd build && cmake .. && make
#RUN mv  /setup.py /warp-ctc/pytorch_binding/ && cd /warp-ctc/pytorch_binding && python setup.py install

#RUN apt-get install python3-libnvinfer-dev -y


