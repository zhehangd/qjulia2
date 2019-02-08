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

#ifndef QJULIA_EFLOAT_H_
#define QJULIA_EFLOAT_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

#include "base.h"

namespace qjulia {

/** \brief Convert a float to a 32-bit unsigned integer
*/
inline BinaryFloat FloatToBinary(Float v) {
  BinaryFloat b;
  std::memcpy(&b, &v, sizeof(Float));
  return b;
}

/** \brief Convert a 32-bit unsigned integer to a float
*/
inline Float BinaryToFloat(BinaryFloat b) {
  Float v;
  std::memcpy(&v, &b, sizeof(BinaryFloat));
  return v;
}

/** \brief Get the next greater float number
*/
inline Float NextFloatUp(Float v) {
  if (std::isinf(v) && v > 0.f) {return v;}
  if (v == -0.f) {v = 0.f;}
  auto b = FloatToBinary(v);
  (v >= 0) ? ++b : --b;
  return BinaryToFloat(b);
}

/** \brief Get the next lesser float number
*/
inline Float NextFloatDown(Float v) {
  if (std::isinf(v) && v < 0.f) {return v;}
  if (v == 0.f) {v = -0.f;}
  auto b = FloatToBinary(v);
  (v >= 0) ? --b : ++b;
  return BinaryToFloat(b);
}

struct EFloat {
  
  inline EFloat(Float f);
  inline EFloat(Float f, Float e);
  
  void Check(void) const;
  
  inline EFloat& operator+=(const EFloat &f);
  inline EFloat& operator-=(const EFloat &f);
  inline EFloat& operator*=(const EFloat &f);
  inline EFloat& operator/=(const EFloat &f);
  
  
  inline EFloat operator+(const EFloat &f) const;
  inline EFloat operator-(const EFloat &f) const;
  inline EFloat operator*(const EFloat &f) const;
  inline EFloat operator/(const EFloat &f) const;
  
  Float Get(void) const {return val;}
  void Set(Float f, float e) {val = f; upper = f + e; lower = f - e;}
  Float Lower(void) const {return lower;}
  Float Upper(void) const {return upper;}
  
  Float val = 0;
  Float upper = 0;
  Float lower = 0;
};

EFloat::EFloat(Float f) : EFloat(f, 0) {
}

EFloat::EFloat(Float f, Float e)
  : val(f), upper(NextFloatUp(f + e)), lower(NextFloatDown(f - e)) {
}

inline std::ostream& operator<<(std::ostream &os, const EFloat f) {
  os << '(' << f.Get() << " [" << f.Lower() << ", " << f.Upper() << "])";
  return os;
}

EFloat& EFloat::operator+=(const EFloat &f) {
  val += f.val;
  upper = NextFloatUp(upper + f.upper);
  lower = NextFloatDown(lower + f.lower);
  return *this;
}

EFloat& EFloat::operator-=(const EFloat &f) {
  val -= f.val;
  upper = NextFloatUp(upper - f.lower);
  lower = NextFloatDown(lower - f.upper);
  return *this;
}

EFloat& EFloat::operator*=(const EFloat &f) {
  val *= f.val;
  Float c1 = lower * f.lower;
  Float c2 = upper * f.lower;
  Float c3 = lower * f.upper;
  Float c4 = upper * f.upper;
  lower = NextFloatDown(std::min(std::min(c1, c2), std::min(c3, c4)));
  upper = NextFloatUp(std::max(std::max(c1, c2), std::max(c3, c4)));
  return *this;
}

EFloat& EFloat::operator/=(const EFloat &f) {
  val /= f.val;
  if (f.lower < 0 && f.upper > 0) {
    lower = kNInf;
    upper = kInf;
  } else {
    Float c1 = lower / f.lower;
    Float c2 = upper / f.lower;
    Float c3 = lower / f.upper;
    Float c4 = upper / f.upper;
    lower = std::min(std::min(c1, c2), std::min(c3, c4));
    upper = std::max(std::max(c1, c2), std::max(c3, c4));
  }
  return *this;
}

EFloat EFloat::operator+(const EFloat &f) const {
  return EFloat(*this) += f;
}

EFloat EFloat::operator-(const EFloat &f) const {
  return EFloat(*this) -= f;
}

EFloat EFloat::operator*(const EFloat &f) const {
  return EFloat(*this) *= f;
}

EFloat EFloat::operator/(const EFloat &f) const {
  return EFloat(*this) /= f;
}



}

#endif
