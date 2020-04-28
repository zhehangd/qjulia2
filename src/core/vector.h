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

#ifndef QJULIA_VECTOR_CUDA_H_
#define QJULIA_VECTOR_CUDA_H_

#include <cmath>
#include <cstring>
#include "base.h"

namespace qjulia {

/** \brief Template base class that implements a fixed length vector

This template class implements access methods, basic mathematical
operations, and a streaming method. This class is used to represent
n-dimentional vectors/coordinates, spectrum, etc. It is also derived
to represent complex numbers, and quaternions.

@tparam T data type
@tparam C the dimension
*/
template<typename T, int C>
class Vec_ {
 public:
  CPU_AND_CUDA Vec_(void) {}
  CPU_AND_CUDA Vec_(T c) {for (int i = 0; i < C; ++i) {vals[i] = c;}}
  CPU_AND_CUDA Vec_(T v0, T v1) {v(0) = v0; v(1) = v1;}
  CPU_AND_CUDA Vec_(T v0, T v1, T v2) {v(0) = v0; v(1) = v1; v(2) = v2;}
  CPU_AND_CUDA Vec_(T v0, T v1, T v2, T v3) {assert(C >= 4); v(0) = v0; v(1) = v1; v(2) = v2; v(3) = v3;}
  
  CPU_AND_CUDA Vec_(const Vec_<T, C> &src) {std::memcpy(vals, src.vals, sizeof(T) * C);}
  
  CPU_AND_CUDA const T& operator[](int i) const {return v(i);}
  CPU_AND_CUDA T& operator[](int i) {return v(i);}
  CPU_AND_CUDA const T& operator()(int i) const {return v(i);}
  CPU_AND_CUDA T& operator()(int i) {return v(i);}
  
  CPU_AND_CUDA const T& v(int i) const {return vals[i];} // for internal access
  CPU_AND_CUDA T& v(int i) {return vals[i];}
  
  CPU_AND_CUDA Vec_<T, C> operator-(void) const;
  
  CPU_AND_CUDA Vec_<T, C>& operator+=(const Vec_<T, C> &p);
  CPU_AND_CUDA Vec_<T, C>& operator-=(const Vec_<T, C> &p);
  
  CPU_AND_CUDA Vec_<T, C> operator+(const Vec_<T, C> &p) const;
  CPU_AND_CUDA Vec_<T, C> operator-(const Vec_<T, C> &p) const;
  
  CPU_AND_CUDA Vec_<T, C>& operator*=(const Vec_<T, C> &p);
  CPU_AND_CUDA Vec_<T, C>& operator/=(const Vec_<T, C> &p);
  
  CPU_AND_CUDA Vec_<T, C> operator*(const Vec_<T, C> &p) const;
  CPU_AND_CUDA Vec_<T, C> operator/(const Vec_<T, C> &p) const;
  
  CPU_AND_CUDA Vec_<T, C>& operator+=(T c);
  CPU_AND_CUDA Vec_<T, C>& operator-=(T c);
  
  CPU_AND_CUDA Vec_<T, C> operator+(T c) const;
  CPU_AND_CUDA Vec_<T, C> operator-(T c) const;
  
  CPU_AND_CUDA Vec_<T, C>& operator*=(T c);
  CPU_AND_CUDA Vec_<T, C>& operator/=(T c);
  
  CPU_AND_CUDA Vec_<T, C> operator*(T c) const;
  CPU_AND_CUDA Vec_<T, C> operator/(T c) const;
  
  CPU_AND_CUDA bool operator==(const Vec_<T, C> &p) const;
  CPU_AND_CUDA bool operator!=(const Vec_<T, C> &p) const {return !(*this == p);}
  
  CPU_AND_CUDA void Fill(T c) const {for (int i = 0; i < C; ++i) {vals[i] = c;}}
  
  /** \brief The square of the L2 norm
  */
  CPU_AND_CUDA Float Norm2(void) const;
  
  /** \brief The L2 norm
  */
  CPU_AND_CUDA Float Norm(void) const;
  
  //T vals[C] = {}; // data
  T vals[C] = {};
};

// - v
template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator-(void) const {
  Vec_<T, C> neg;
  for (int i = 0; i < C; ++i) {neg.v(i) = - v(i);}
  return neg;
}

// v1 += v2
template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator+=(T c) {
  for (int i = 0; i < C; ++i) {v(i) += c;}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator-=(T c) {
  for (int i = 0; i < C; ++i) {v(i) -= c;}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator+(T c) const {
  return Vec_(*this) += c;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator-(T c) const {
  return Vec_(*this) -= c;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator*=(T c) {
  for (int i = 0; i < C; ++i) {v(i) *= c;}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator/=(T c) {
  for (int i = 0; i < C; ++i) {v(i) /= c;}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator*(T c) const {
  return Vec_(*this) *= c;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator/(T c) const {
  return Vec_(*this) /= c;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator+=(const Vec_<T, C> &p) {
  for (int i = 0; i < C; ++i) {v(i) += p.v(i);}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator-=(const Vec_<T, C> &p) {
  for (int i = 0; i < C; ++i) {v(i) -= p.v(i);}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator+(const Vec_<T, C> &p) const {
  return Vec_(*this) += p;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator-(const Vec_<T, C> &p) const {
  return Vec_(*this) -= p;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator*=(const Vec_<T, C> &p) {
  for (int i = 0; i < C; ++i) {v(i) *= p.v(i);}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C>& Vec_<T, C>::operator/=(const Vec_<T, C> &p) {
  for (int i = 0; i < C; ++i) {v(i) /= p.v(i);}
  return *this;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator*(const Vec_<T, C> &p) const {
  return Vec_(*this) *= p;
}

template<typename T, int C> inline
CPU_AND_CUDA Vec_<T, C> Vec_<T, C>::operator/(const Vec_<T, C> &p) const {
  return Vec_(*this) /= p;
}

template<typename T, int C> inline
CPU_AND_CUDA bool Vec_<T, C>::operator==(const Vec_<T, C> &p) const {
  for (int i = 0; i < C; ++i) {
    if (v(i) != p.v(i)) {return false;}
  }
  return true;
}

template<typename T, int C> inline
CPU_AND_CUDA Float Vec_<T, C>::Norm2(void) const {
  Float sum = 0;
  for (int i = 0; i < C; ++i) {
    Float v = this->v(i);
    sum += v * v;
  }
  return sum;
}

template<typename T, int C> inline
CPU_AND_CUDA Float Vec_<T, C>::Norm(void) const {
  return std::sqrt(Norm2());
}

/** \brief The square of the L2 Distance between two points
*/
template<typename T, int C>
CPU_AND_CUDA Float Dist2(const Vec_<T, C> &p1, const Vec_<T, C> &p2) {
  return (p1 - p2).Norm2();
}

/** \brief The L2 Distance between two points
*/
template<typename T, int C>
CPU_AND_CUDA Float Dist(const Vec_<T, C> &p1, const Vec_<T, C> &p2) {
  return std::sqrt(Dist2(p1, p2));
}

template<typename T, int C>
CPU_AND_CUDA bool IsFinite(const Vec_<T, C> &vec) {
  for (auto &v : vec.vals) {if (!std::isfinite(v)) {return false;}}
  return true;
}

template<typename T, int C>
CPU_AND_CUDA Vec_<T, C> Clamp(const Vec_<T, C> &vec, T min_v, T max_v) {
  Vec_<T, C> vec_out;
  for (int i = 0; i < C; ++i) {
    T v = vec[i];
    if (v > max_v) {
      v = max_v;
    } else if (v < min_v) {
      v = min_v;
    }
    vec_out[i] = v;
  }
  return vec_out;
}


/** \brief Dump vector to stream

This is made compatible with the scene description format.
*/
template<typename T, int C> inline
std::ostream& operator<<(std::ostream &os, const Vec_<T, C> &p) {
  for (int i = 0; i < C; ++i) {
    os << (p[i] + 0); // +0 is a trick to avoid showing characters when T=char
    if (i != (C - 1)) {os << ',';}
  }
  return os;
}

/** \brief Read vector from stream

This is made compatible with the scene description format.
*/
template<typename T, int C> inline
std::istream& operator>>(std::istream &iss, Vec_<T, C> &vec) {
  for (int i = 0; i < C; ++i) {
    iss >> vec[i];
    if (iss.fail()) {return iss;}
    if (i == (C - 1)) {return iss;}
    iss.ignore(1);
  }
  return iss;
}

template<typename T, int C>
CPU_AND_CUDA Vec_<T, C> multiply(const Vec_<T, C> &p1, const Vec_<T, C> &p2) {
  Vec_<T, C> v;
  for (int i = 0; i < C; ++i) {v[i] = p1[i] * p2[i];}
  return v;
}

/** \brief Dot product
*/
template<typename T, int C>
CPU_AND_CUDA Float Dot(const Vec_<T, C> &p1, const Vec_<T, C> &p2) {
  Float sum = 0;
  for (int i = 0; i < C; ++i) {sum += p1(i) * p2(i);}
  return sum;
}

template<typename T>
CPU_AND_CUDA Vec_<Float, 3> Cross(const Vec_<T, 3> &p1, const Vec_<T, 3> &p2) {
  return Vec_<Float, 3>(
    p1(1) * p2(2) - p1(2) * p2(1), p1(2) * p2(0) - p1(0) * p2(2),
    p1(0) * p2(1) - p1(1) * p2(0));
}

// Project p1 to p2.
template<typename T, int C>
CPU_AND_CUDA Vec_<Float, C> Project(const Vec_<T, C> &p1, const Vec_<T, C> &p2) {
  return p2 * (Dot(p1, p2) / p2.Norm2());
}

template<typename T, int C>
CPU_AND_CUDA Vec_<Float, C> Normalize(const Vec_<T, C> &p) {
  return p / p.Norm();
}

typedef std::uint8_t Byte;
typedef Vec_<int, 2> Point2i;
typedef Vec_<Float, 2> Point2f;
typedef Vec_<Float, 3> Point3f;
typedef Vec_<Float, 4> Point4f;
typedef Vec_<Byte, 3> Vector3b;
typedef Vec_<Byte, 4> Vector4b;
typedef Vec_<int, 2> Vector2i;
typedef Vec_<Float, 2> Vector2f;
typedef Vec_<Float, 3> Vector3f;
typedef Vec_<Float, 4> Vector4f;
typedef Vec_<unsigned char, 3> Pixel;

/** \brief Class for complex numbers
*/
class Complex : public Vec_<Float, 2> {
 public:
  Complex(void) {}
  Complex(Float a, Float b) : Vec_<Float, 2>(a, b) {}
  Complex(const Vec_<Float, 2> &p) : Vec_<Float, 2>(p) {}
  
  const Float& Real(void) const {return v(0);}
  const Float& Imag(void) const {return v(1);}
  Float& Real(void) {return v(0);}
  Float& Imag(void) {return v(1);}
  
  Complex Conj(void) const {return Complex(v(0), -v(1));}
  
  Complex& operator*=(Float c);
  Complex& operator/=(Float c);
  
  Complex operator*(Float c) const;
  Complex operator/(Float c) const;
  
  Complex& operator*=(const Complex &p);
  Complex operator*(const Complex &p) const;
  
  Complex& operator/=(const Complex &p) = delete;
  Complex operator/(const Complex &p) const = delete;
};

inline Complex& Complex::operator*=(Float  c) {
  for (int i = 0; i < 2; ++i) {v(i) *= c;}
  return *this;
}

inline Complex& Complex::operator/=(Float c) {
  for (int i = 0; i < 2; ++i) {v(i) /= c;}
  return *this;
}

inline Complex Complex::operator*(Float c) const {
  return Complex(*this) *= c;
}


inline Complex Complex::operator/(Float c) const {
  return Complex(*this) /= c;
}

inline Complex& Complex::operator*=(const Complex &p) {
  Float real = Real() * p.Real() - Imag() * p.Imag();
  Float imag = Real() * p.Imag() + Imag() * p.Real();
  Real() = real;
  Imag() = imag;
  return *this;
}

inline Complex Complex::operator*(const Complex &p) const {
  return Complex(*this) *= p;
}

/** \brief Class for quaterions
*/
class Quaternion : public Vec_<Float, 4> {
 public:
  CPU_AND_CUDA Quaternion(void) {}
  CPU_AND_CUDA Quaternion(Float a, Float b, Float c, Float d) : Vec_<Float, 4>(a, b, c, d) {}
  CPU_AND_CUDA Quaternion(const Vec_<Float, 4> &p) : Vec_<Float, 4>(p) {}
  
  CPU_AND_CUDA const Float& Real(void) const {return v(0);}
  CPU_AND_CUDA const Float& ImagI(void) const {return v(1);}
  CPU_AND_CUDA const Float& ImagJ(void) const {return v(2);}
  CPU_AND_CUDA const Float& ImagK(void) const {return v(3);}
  CPU_AND_CUDA Float& Real(void) {return v(0);}
  CPU_AND_CUDA Float& ImagI(void) {return v(1);}
  CPU_AND_CUDA Float& ImagJ(void) {return v(2);}
  CPU_AND_CUDA Float& ImagK(void) {return v(3);}
  
  CPU_AND_CUDA Quaternion Conj(void) const {return Quaternion(v(0), -v(1), -v(2), -v(3));}
  
  CPU_AND_CUDA Quaternion& operator*=(const Quaternion &p);
  CPU_AND_CUDA Quaternion operator*(const Quaternion &p) const;
  
  CPU_AND_CUDA Quaternion& operator/=(const Quaternion &p) = delete;
  CPU_AND_CUDA Quaternion operator/(const Quaternion &p) const = delete;
  
  CPU_AND_CUDA Quaternion& operator*=(Float c);
  CPU_AND_CUDA Quaternion operator*(Float c) const;
  
  CPU_AND_CUDA Quaternion& operator/=(Float c) = delete;
  CPU_AND_CUDA Quaternion operator/(Float c) const = delete;
};

CPU_AND_CUDA inline Quaternion& Quaternion::operator*=(const Quaternion &p) {
  Float a = Real() * p.Real() - ImagI() * p.ImagI()
    - ImagJ() * p.ImagJ() - ImagK() * p.ImagK();     
  Float i = Real() * p.ImagI() + ImagI() * p.Real()
    + ImagJ() * p.ImagK() - ImagK() * p.ImagJ();
  Float j = Real() * p.ImagJ() - ImagI() * p.ImagK()
    + ImagJ() * p.Real() + ImagK() * p.ImagI();
  Float k = Real() * p.ImagK() + ImagI() * p.ImagJ()
    - ImagJ() * p.ImagI() + ImagK() * p.Real();
  Real() = a;
  ImagI() = i;
  ImagJ() = j;
  ImagK() = k;
  return *this;
}

CPU_AND_CUDA inline Quaternion Quaternion::operator*(const Quaternion &p) const {
  return Quaternion(*this) *= p;
}

CPU_AND_CUDA inline Quaternion& Quaternion::operator*=(Float c) {
  for (int i = 0; i < 4; ++i) {v(i) *= c;}
  return *this;
}

CPU_AND_CUDA inline Quaternion Quaternion::operator*(Float c) const {
  return Quaternion(*this) *= c;
}

}

#endif
