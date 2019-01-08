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
