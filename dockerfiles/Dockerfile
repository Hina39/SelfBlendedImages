FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8"

RUN apt -y update && \
    apt -y install python3

RUN apt update && apt upgrade -y && \
    apt install -y cmake git screen\
    build-essential python3-dev libopencv-dev
RUN apt install -y libgl1-mesa-dev \
    python3-pip


RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install dlib imutils scipy scikit-learn pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch&&\
    pip3 install -U retinaface_pytorch &&\
    apt clean

WORKDIR /app 