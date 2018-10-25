#pragma once
#ifdef HAVE_GLOG
// use glog
#else
#include <tensorflow/core/framework/op.h>
#endif
