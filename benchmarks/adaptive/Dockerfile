FROM ubuntu:disco

ADD docker/sources.list.disco-aliyun /etc/apt/sources.list

RUN apt update \
    && apt install -y build-essential cmake python3 python3-pip golang-1.12-go \
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.13.1

WORKDIR /src/kungfu
ENV PATH /usr/lib/go-1.12/bin:/src/kungfu/bin:${PATH}

ADD . .

RUN pip3 install --no-index -U -v .
RUN ./configure --build-tools && make
