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

#include <glog/logging.h>

namespace qjulia {

typedef int SizeType;
  
typedef float Float;

typedef std::numeric_limits<Float> FloatLimits;

typedef std::uint32_t BinaryFloat;

static constexpr Float kEpsilon = std::numeric_limits<Float>::epsilon() * 0.5f;

constexpr Float kInf = std::numeric_limits<Float>::infinity();

constexpr Float kNInf = - std::numeric_limits<Float>::infinity();

constexpr Float kPi = 3.1415926f;

inline Float Degree2Rad(Float d) {return d * kPi / 180.0f;}

inline constexpr Float kGamma(int n) {
  return (n * kEpsilon) / (1 - n * kEpsilon);
}

enum class EntityType {
  kObject = 0,
  kShape = 1,
  kTransform = 2,
  kMaterial = 3,
  kLight = 4,
  kCamera = 5,
  kScene = 6,
};

const int kNumEntityTypes = 7;

const char * const kEntityTypeNames[] = {
  "object", "shape", "transform", "material", "light", "camera", "scene"};

inline int GetEntityTypeID(EntityType type) {
  return static_cast<int>(type);
}

inline const char* GetEntityTypeName(EntityType type) {
  return kEntityTypeNames[GetEntityTypeID(type)];
}

}

#endif
