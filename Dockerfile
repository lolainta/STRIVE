FROM nvidia/cuda:12.1.0-base-ubuntu18.04

RUN apt update
RUN apt install -y python3 python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
RUN pip3 install https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
RUN pip3 install https://download.pytorch.org/whl/torchaudio-0.9.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.7-cp36-cp36m-linux_x86_64.whl
RUN pip3 install https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_sparse-0.6.10-cp36-cp36m-linux_x86_64.whl

ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcusparse-11-0_11.1.1.245-1_amd64.deb .
RUN dpkg -i /libcusparse-11-0_11.1.1.245-1_amd64.deb
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib/

COPY requirements.txt .
RUN MAKEFLAGS="-j$(nproc)" pip3 install -r requirements.txt
# RUN pip3 install torch_geometric
# RUN pip3 install tqdm
# RUN pip3 install matplotlib
# RUN pip3 install configargparse

# RUN dpkg -i 
# RUN pip3 install nuscenes-devkit==1.1.5

WORKDIR /strive
# RUN pip3 install -r requirements.txt

CMD sh

