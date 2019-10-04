# KungFu Contributor Guideline

## Intro

KungFu is a distributed machine learning framework implemented in Golang and C++.
It also provides Python binding for TensorFlow.

KungFu can be integrated into existing TensorFlow programs as easy as Horovod.
And it also be used as a standalone C/C++/Golang library if you are building
your own machine learning framework.

## Requirements

* Golang1.11 or above is required (For the [Modules](https://github.com/golang/go/wiki/Modules) feature).
* CMake is required for building the C++ sources.
* Python and TensorFlow is required if you are going to build the TensorFlow binding.
* gtest is used for unittest, it can be auto fetched and built from source.

## Project Structure

All source code are under `./srcs/<lang>/` where `<lang> := cpp | go | python`.

### Components

* kungfu-base: a C library that will be used by Golang via CGO internally
* kungfu-comm: a C library implemented in Golang
* kungfu: the public C/C++ library based on kungfu-comm

### Concepts

* PeerID: PeerID is the unique identifier of a peer, it tells the runner how to start the peer and also tells all peers how to find each other.

* PeerList: PeerList is the ordered list of **PeerID** from all peers in the cluster. It is a common constant shared among all peers in the cluster.

* HostSpec: HostSpec is the metadata that describes a host machine.

* Graph: A directed graph, which may contain self loops. The vertices are numbered from 0 to n - 1.
