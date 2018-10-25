# docker build --rm -t kungfu-builder -f docker/Dockerfile .
# docker run --rm -it kungfu-builder
FROM ubuntu:bionic AS builder

# ADD docker/sources.list.ustc /etc/apt/sources.list
RUN apt update && \
    apt install -y build-essential cmake python3 python3-pip && \
    pip3 install tensorflow

FROM builder

ADD . /kungfu
WORKDIR /kungfu
