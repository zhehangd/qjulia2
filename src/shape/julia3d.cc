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

#include "julia3d.h"

#include <vector>
#include <memory>

#include "core/vector.h"
#include "core/shape.h"
#include "core/algorithm.h"
#include "core/resource_mgr.h"

namespace qjulia {


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

void Julia3DShape::Iterate(Quaternion &q, Quaternion &qp) const {
  qp *= q;
  qp *= (Float)2;
  q *= q;
  q += constant_;
}

void Julia3DShape::Iterate(Quaternion &q, Quaternion &qp, int n) const {
  for (int i = 0; i < n; ++i) {Iterate(q, qp);}
}

void Julia3DShape::Iterate(Quaternion &q) const {
  q *= q;
  q += constant_;
}

void Julia3DShape::Iterate(Quaternion &q, int n) const {
  for (int i = 0; i < n; ++i) {Iterate(q);}
}

Vector3f Julia3DShape::EstimateNormal(const Vector3f &v) const {
  Float eps = 1e-3;
  Quaternion q(v[0], v[1], v[2], 0);
  std::vector<Quaternion> neighbors(6);
  
  neighbors[0] = q - Quaternion(eps, 0, 0, 0);
  neighbors[1] = q + Quaternion(eps, 0, 0, 0);
  neighbors[2] = q - Quaternion(0, eps, 0, 0);
  neighbors[3] = q + Quaternion(0, eps, 0, 0);
  neighbors[4] = q - Quaternion(0, 0, eps, 0);
  neighbors[5] = q + Quaternion(0, 0, eps, 0);
  
  // NOTE: This implementatiuon is not very stable,
  // with large 'niters' the value goes to inf.
  // May replace this with a more stable method.
  for (int k = 0; k < (int)neighbors.size(); ++k) {
    int niters = 5;
    Quaternion &nq = neighbors[k];
    for (int i = 0; i < niters; ++i) {
      Iterate(nq);
    }
    assert(IsFinite(nq));
  }
  
  Float nx = neighbors[1].Norm() - neighbors[0].Norm();
  Float ny = neighbors[3].Norm() - neighbors[2].Norm();
  Float nz = neighbors[5].Norm() - neighbors[4].Norm();
  
  Vector3f normal = Normalize(Vector3f(nx, ny, nz));
  return normal;
}

int Julia3DShape::TestFractal(Quaternion &q, Quaternion &qp) const {
  float norm = 0;
  for (int n = 0; n < max_iterations_; ++n) {
    Iterate(q, qp);
    norm = q.Norm();
    if (norm > max_magnitude_) {return n;}
  }
  return max_iterations_;
}

int Julia3DShape::TestFractal(Quaternion &q) const {
  float norm = 0;
  for (int n = 0; n < max_iterations_; ++n) {
    Iterate(q);
    norm = q.Norm();
    if (norm > max_magnitude_) {return n;}
  }
  return max_iterations_;
}

Julia3DShape::FractalTestRet Julia3DShape::SearchIntersection(
    const Vector3f &start, const Vector3f &dir, Float max_dist) const {
  FractalTestRet ret;
  Vector3f p = start;
  while((p - start).Norm() < max_dist) {
    Quaternion q(p[0], p[1], p[2], 0);
    Quaternion qp(1, 0, 0, 0);
    TestFractal(q, qp);
    // Esitmate distance.
    Float q_norm = q.Norm(); // TODO: cover 0
    Float qp_norm = qp.Norm();
    Float d = 0.5 * q_norm * std::log(q_norm) / qp_norm;
    if (d < 1e-3) {
      ret.has_intersection = true;
      ret.isect_position = p;
      ret.dist = (p - start).Norm();
      return ret;
    }
    p += dir * d;
  }
  return ret;
}

Intersection Julia3DShape::Intersect(const Vector3f &start, const Vector3f &dir) const {
  Intersection isect;
  
  Float tl, tg;
  if (!IntersectSphere(start, dir, bounding_radius_, &tl, &tg) || tg < 0) {return isect;}
  tl = std::max((Float)0, tl);
  
  //
  Vector3f bound_start = start + dir * tl;
  FractalTestRet ret = SearchIntersection(bound_start, dir, tg - tl);
  if (!ret.has_intersection) {return isect;}
  isect.good = true;
  isect.dist = tl + ret.dist;
  isect.position = ret.isect_position;
  isect.normal = EstimateNormal(isect.position);
  return isect;
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
