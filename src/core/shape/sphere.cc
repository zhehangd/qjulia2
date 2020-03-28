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

#include "core/shape/sphere.h"

#include <vector>
#include <memory>

#include "core/arg_parse.h"
#include "core/algorithm.h"
#include "core/shape.h"
#include "core/vector.h"

namespace qjulia {

CPU_AND_CUDA Intersection SphereShape::Intersect(const Ray &ray) const {
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

void SphereShape::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  if (args.size() == 0) {return;}
  if (args[0] == "SetPosition") {
    ParseArg(args[1], position);
  } else if (args[0] == "SetRadius") {
    ParseArg(args[1], radius);
  } else {
    throw UnknownCommand(args[0]);
  }
}

}
