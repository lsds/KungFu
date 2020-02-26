FROM ubuntu:bionic

# ADD docker/sources.list.aliyun /etc/apt/sources.list
RUN apt update && apt install -y software-properties-common

# https://github.com/golang/go/wiki/Ubuntu
RUN add-apt-repository ppa:longsleep/golang-backports

RUN apt update \
    && apt install -y build-essential cmake python3 python3-pip wget git \
        golang-1.13-go

# RUN wget -q https://dl.google.com/go/go1.11.linux-amd64.tar.gz \
#     && tar -C /usr/local -xf go1.11.linux-amd64.tar.gz \
#     && rm go1.11.linux-amd64.tar.gz

# -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install numpy==1.16 tensorflow==1.13.2

ENV GOROOT /usr/lib/go-1.13

WORKDIR /src/kungfu
ENV PATH ${GOROOT}/bin:${PATH}
ENV TF_CPP_MIN_LOG_LEVEL 2

ADD . .

RUN pip3 install --no-index -U -v .
