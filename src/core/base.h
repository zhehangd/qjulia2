/*

MIT License

Copyright (c) 2019 Zhehang Ding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef QJULIA_QJULIA_H_
#define QJULIA_QJULIA_H_

#include <cassert>
#include <cstdint>
#include <limits>

#include <fmt/format.h>
#include <glog/logging.h>

#include "core/config.h"

#ifdef __CUDACC__
#define CPU_AND_CUDA __device__ __host__
#define KERNEL __global__
#else
#define CPU_AND_CUDA
#define KERNEL
#endif

namespace qjulia {

typedef int SizeType;
  
typedef float Float;

typedef std::numeric_limits<Float> FloatLimits;

typedef std::uint32_t BinaryFloat;

static constexpr Float kEpsilon = std::numeric_limits<Float>::epsilon() * 0.5f;

constexpr Float kInf = std::numeric_limits<Float>::infinity();

constexpr Float kNInf = - std::numeric_limits<Float>::infinity();

constexpr Float kPi = 3.1415926f;

CPU_AND_CUDA inline Float Degree2Rad(Float d) {return d * kPi / 180.0f;}

CPU_AND_CUDA inline constexpr Float kGamma(int n) {
  return (n * kEpsilon) / (1 - n * kEpsilon);
}

#ifdef __CUDACC__
inline void CUDACheckError(int line, cudaError_t err) {
  if (err == cudaSuccess) {return;}
  LOG(FATAL) << "Line #" << line << " " << cudaGetErrorName(err) << ":" << cudaGetErrorString(err);
};
#endif

}

#endif
