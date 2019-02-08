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

#include "qjulia2/shape/plane.h"

#include <vector>
#include <memory>

#include "qjulia2/core/vector.h"
#include "qjulia2/core/shape.h"
#include "qjulia2/core/algorithm.h"
#include "qjulia2/core/resource_mgr.h"

namespace qjulia {

PlaneShape::PlaneShape(void) : PlaneShape({}, {0, 0, 1}) {
}

PlaneShape::PlaneShape(Vector3f position, Vector3f orientation) {
  SetPositionAndNormal(position, orientation);
}

void PlaneShape::SetPositionAndNormal(Vector3f position, Vector3f orientation) {
  normal = Normalize(orientation);
  offset = - Dot(position, normal);
}

Intersection PlaneShape::Intersect(const Ray &ray) const {
  Intersection isect;
  Float num = - (Dot(normal, ray.start) + offset);
  Float den = Dot(normal, ray.dir);
  Float dist;
  if (std::isinf(num) || std::isinf(den)) {
    dist = std::signbit(num) == std::signbit(den) ? kInf : kNInf;
  } else {
    dist = num / den;
  }
  if (dist >= 0) {
    isect.good = true;
    isect.dist = dist;
    isect.position = ray.start + ray.dir * dist;
    isect.normal = normal;
  }
  return isect;
}

bool PlaneShape::ParseInstruction(
    const TokenizedStatement instruction, 
    const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  if (instruction[0] == "set") {
    Vector3f v2[2];
    bool good = ParseInstruction_Value<Vector3f, 2>(instruction, resource, v2);
    if (good) {SetPositionAndNormal(v2[0], v2[1]);}
    return good;
  } else {
    return UnknownInstructionError(instruction);
  }
}


}
