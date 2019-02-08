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

#include "qjulia2/shape/sphere.h"

#include <vector>
#include <memory>

#include "qjulia2/core/vector.h"
#include "qjulia2/core/shape.h"
#include "qjulia2/core/algorithm.h"
#include "qjulia2/core/resource_mgr.h"

namespace qjulia {

Intersection SphereShape::Intersect(const Ray &ray) const {
  Intersection isect;
  Float tl, tg;
  Vector3f start = ray.start - position;
  bool has_root = IntersectSphere(start, ray.dir, radius, &tl, &tg);
  if (has_root && tg >= 0) {
    isect.good = true;
    isect.dist = tl > 0 ? tl : tg;
    isect.position = ray.start + ray.dir * isect.dist;
    isect.normal = Normalize(isect.position - position);
  }
  return isect;
}

bool SphereShape::ParseInstruction(
    const TokenizedStatement instruction, 
    const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  if (instruction[0] == "position") {
    return ParseInstruction_Value<Vector3f>(instruction, resource, &position);
  } else if (instruction[0] == "radius") {
    return ParseInstruction_Value<Float>(instruction, resource, &radius);
  } else {
    return UnknownInstructionError(instruction);
  }
}


}
