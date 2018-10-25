#pragma once
#include <cstddef>
#include <string>

#include "message.h"

/*!
An agent is a worker that can accepts gradients from local worker or other
peers, and provide gradient to local workers.
*/
class Agent
{
  public:
    /*! Accept gradient from local worker. */
    virtual void push(const std::string &name, const void *data, size_t n) = 0;

    /*! Accept gradient from peers. */
    virtual void recv(const std::string &name, const void *data, size_t n) = 0;

    /*! Provide gradient to local worker. */
    virtual void pull(const std::string &name, void *data, size_t n) = 0;

    static Agent *get_instance();
};
