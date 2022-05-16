FROM pytorch/pytorch
RUN apt-get update
USER root

COPY main.py .
COPY nvprof.sh .
COPY nsys.sh .