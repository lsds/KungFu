# docker build --rm -t kungfu-builder -f docker/Dockerfile .
# docker run --rm -it kungfu-builder
FROM ubuntu:bionic

# ADD docker/sources.list.ustc /etc/apt/sources.list
RUN apt update && \
    apt install -y make cmake g++ python3 python3-pip && \
    pip3 install tensorflow

ADD . /kungfu

WORKDIR /kungfu
