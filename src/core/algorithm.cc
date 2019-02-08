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

#include "qjulia2/core/algorithm.h"

#include <algorithm>
#include <cmath>

namespace qjulia {

bool SolveQuadratic(Float a, Float b, Float c, Float *tl, Float *tg) {
  // TODO This is a naive implementation which is not stable
  // when a is close to 0. Improvement is needed.
  Float d = b * b - 4 * a * c;
  if (d < 0) {
    return false;
  } else {
    d = std::sqrt(d);
    *tl = (-b + d) / (2 * a);
    *tg = (-b - d) / (2 * a);
    if (*tl > *tg) {std::swap(*tl, *tg);}
    return true;
  }
}

bool IntersectSphere(const Vector3f start, const Vector3f dir,
                      Float r, Float *tl, Float *tg) {
  Float a = dir.Norm2();
  Float b = 2 * Dot(start, dir);
  Float c = start.Norm2() - r * r;
  bool has_root = SolveQuadratic(a, b, c, tl, tg);
  return has_root;
}

}
