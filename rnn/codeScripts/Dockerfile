
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

USER root
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 
RUN apt-get update && apt-get install --assume-yes git && pip3 install -U torchtext==0.12.0 && pip install -U pip setuptools wheel\
&& pip install -U spacy && python -m spacy download en_core_web_sm

RUN apt-get update