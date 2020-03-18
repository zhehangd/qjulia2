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

#include "core/shape/julia3d.h"

#include <vector>
#include <memory>

#include "core/vector.h"
#include "core/shape.h"
#include "core/algorithm.h"
#include "core/resource_mgr.h"

#include "julia3d_impl_cpu.h"
#include "julia3d_impl_gpu.h"

namespace qjulia {

Intersection Julia3DShape::Intersect(const Ray &ray) const {
  Intersection isect;
  Julia3DIntersectCPU(
    ray, isect, constant_, max_iterations_,
    max_magnitude_, bounding_radius_);
  return isect;
}

void Julia3DShape::Intersect(const Array2D<Ray> &rays,
                 Array2D<Intersection> &isects) const {
  Julia3DIntersectGPU(
    rays, isects, constant_, max_iterations_,
    max_magnitude_, bounding_radius_);
}
  
void Julia3DShape::UsePreset(int i) {
  Quaternion constant;
  switch (i) {
    case 0: constant = {-1, 0.2, 0, 0}; break;
    case 1: constant = {-0.2, 0.8, 0, 0}; break;
    case 2: constant = {-0.125,-0.256,0.847,0.0895}; break;
    default:;
  }
  SetConstant(constant);
}

bool Julia3DShape::ParseInstruction(
    const TokenizedStatement instruction, 
    const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  const std::string &name = instruction[0];
  if (name == "constant") {
    Vector3f constant;
    bool good = ParseInstruction_Value<Vector3f>(
      instruction, resource, &constant);
    if (good) {
      for (int i = 0; i < 3; ++i) {constant_[i] = constant[i];}
      constant[3] = 0;
    }
    return true;
  } else if (name == "max_iterations") {
    return ParseInstruction_Value<int>(
      instruction, resource, &max_iterations_);
  
  } else if (name == "max_magnitude") {
    return ParseInstruction_Value<Float>(
      instruction, resource, &max_magnitude_);
  
  } else if (name == "bounding_radius") {
    return ParseInstruction_Value<Float>(
      instruction, resource, &bounding_radius_);
  
  } else {
    return UnknownInstructionError(instruction);
  }
}

}
