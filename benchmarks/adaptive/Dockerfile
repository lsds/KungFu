FROM ubuntu:bionic

ADD docker/sources.list.aliyun /etc/apt/sources.list
RUN apt update && apt install -y software-properties-common

# https://github.com/golang/go/wiki/Ubuntu
# RUN add-apt-repository ppa:longsleep/golang-backports # TOO SLOW

RUN apt update \
    && apt install -y build-essential cmake python3 python3-pip wget

RUN wget -q https://dl.google.com/go/go1.11.linux-amd64.tar.gz \
    && tar -C /usr/local -xf go1.11.linux-amd64.tar.gz \
    && rm go1.11.linux-amd64.tar.gz

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.12

WORKDIR /src/kungfu
ENV PATH ${PATH}:/usr/local/go/bin:/src/kungfu/bin
ENV TF_CPP_MIN_LOG_LEVEL 2

ADD . .

RUN ./configure --build-tools --build-tensorflow-ops && make
RUN pip3 install --no-index -U -v .
