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

#ifndef QJULIA_ALGORITHM_H_
#define QJULIA_ALGORITHM_H_

/** \file Common algorithms
*/

#include "base.h"
#include "ray.h"
#include "vector.h"

namespace qjulia {

/** \brief Solve a quadratic equation

Solve a*x^2 + b*x + c = 0 for its real roots. If the equation has real roots,
they are solved in t0 and t1 and true is returned. Otherwise t0 are t1 are
not changed and false is returned.
*/
bool SolveQuadratic(Float a, Float b, Float c, Float *t0, Float *t1);

bool IntersectSphere(const Vector3f start, const Vector3f dir,
                     Float r, Float *tl, Float *tg);

// This function produces a basis of the vector space orthogonal to a unit
// vector 'vec'. The number of basis vectors equals the dimension minus one.
// NOTE: 'vec' must has magnitude one otherwise the result would be incorrect.
// The magnitude of the basis vectors are not necessarily one.
template<typename Float, int C>
std::vector<Vec_<Float, C>> ProduceOrthogonalBasis(const Vec_<Float, C> &vec) {
  // Pick the dimension corresponding to the maximum coordinate of 'vec'.
  const int dim = C;
  int max_dim = 0;
  Float max_v = std::abs(vec[0]);
  for (int i = 1; i < dim; ++i) {
    Float v = std::abs(vec[i]);
    if (v > max_v) {
      max_dim = i;
      max_v = v;
    }
  }
  
  // Generate basis vectors.
  std::vector<Vector3f> basis;
  basis.reserve(dim - 1);
  for (int i = 0; i < dim; ++i) {
    if (i == max_dim) {continue;}
    Vector3f basis_vec;
    basis_vec[i] = 1.0f;
    basis_vec = basis_vec - vec * Dot(vec, basis_vec);
    basis.push_back(basis_vec);
  }
  return basis;
}

}

#endif
