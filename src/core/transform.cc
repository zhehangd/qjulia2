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

#include "transform.h"

#include "resource_mgr.h"

namespace qjulia {

namespace {
#if 0
void MakeIdentity(Matrix4x4 &mat) {
  mat[0] = mat[5] = mat[10] = mat[15] = 1;
  mat[1] = mat[2] = mat[3] = 0;
  mat[4] = mat[6] = mat[7] = 0;
  mat[8] = mat[9] = mat[11] = 0;
  mat[12] = mat[13] = mat[14] = 0;
}
#endif
}

Matrix4x4::Matrix4x4(void) {
  m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
}

Matrix4x4 Matrix4x4::Identity(void) {
  Matrix4x4 mat;
  mat.m[0][0] = mat.m[1][1] = mat.m[2][2] = mat.m[3][3] = 1;
  return mat;
}

Matrix4x4 Matrix4x4::Translate(const Vector3f &T) {
  Matrix4x4 mat;
  for (int i = 0; i < 3; ++i) {
    mat.m[i][3] = T[i];
  }
  return mat;
}

Matrix4x4 Matrix4x4::Scale(const Vector3f &S) {
  Matrix4x4 mat;
  for (int i = 0; i < 3; ++i) {
    mat.m[i][i] = S[i];
  }
  return mat;
}

Matrix4x4 Matrix4x4::RotateX(const Float angle) {
  Float rad = Degree2Rad(angle);
  Float c = std::cos(rad);
  Float s = std::sin(rad);
  Matrix4x4 mat;
  mat.m[1][1] = c;
  mat.m[2][2] = c;
  mat.m[1][2] = -s;
  mat.m[2][1] = s;
  return mat;
}

Matrix4x4 Matrix4x4::RotateY(const Float angle) {
  Float rad = Degree2Rad(angle);
  Float c = std::cos(rad);
  Float s = std::sin(rad);
  Matrix4x4 mat;
  mat.m[0][0] = c;
  mat.m[2][2] = c;
  mat.m[0][2] = s;
  mat.m[2][0] = -s;
  return mat;
}

Matrix4x4 Matrix4x4::RotateZ(const Float angle) {
  Float rad = Degree2Rad(angle);
  Float c = std::cos(rad);
  Float s = std::sin(rad);
  Matrix4x4 mat;
  mat.m[0][0] = c;
  mat.m[1][1] = c;
  mat.m[0][1] = -s;
  mat.m[1][0] = s;
  return mat;
}


Matrix4x4 Matrix4x4::operator*(const Matrix4x4 &mat) const {
  Matrix4x4 o;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      o.m[i][j] = 
        m[i][0] * mat.m[0][j] +
        m[i][1] * mat.m[1][j] +
        m[i][2] * mat.m[2][j] +
        m[i][3] * mat.m[3][j];
    }
  }
  return o;
}

Matrix4x4& Matrix4x4::operator*=(const Matrix4x4 &mat) {
  *this = *this * mat;
  return *this;
}

Vector3f Matrix4x4::MulMatVec(const Vector3f &v, Float w) const {
  Vector3f o;
  for (int i = 0; i < 3; ++i) {
    o[i] = m[i][0] * v[0] + 
           m[i][1] * v[1] + 
           m[i][2] * v[2] + 
           m[i][3] * w;
  }
  return o;
}

Vector3f Matrix4x4::MulTranspMatVecMatVec(const Vector3f &v, Float w) const {
  Vector3f o;
  for (int i = 0; i < 3; ++i) {
    o[i] = m[0][i] * v[0] + 
           m[1][i] * v[1] + 
           m[2][i] * v[2] + 
           m[3][i] * w;
  }
  return o;
}

std::ostream& operator<<(std::ostream &os, const Matrix4x4 &mat) {
  os << '\n';
  for (int i = 0; i < 4; ++i) {
    os << mat.m[i][0] << ", " << mat.m[i][1] << ", "
       << mat.m[i][2] << ", " << mat.m[i][3] << "\n";
  }
  return os;
}

void Transform::Identity(void) {
  mat_wo_.Identity();
  mat_ow_.Identity();
}

Vector3f Transform::W2O_Point(const Vector3f &vec) const {
  return mat_wo_ .MulMatVec(vec, 1);
}

Vector3f Transform::W2O_Vector(const Vector3f &vec) const {
  return mat_wo_ .MulMatVec(vec, 0);
}

Vector3f Transform::W2O_Normal(const Vector3f &vec) const {
  // http://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Applying_Transformations.html#Normals
  // Using mat_ow_ rather than mat_wo_ is not a typo.
  // Transforming a normal is different from transforming a vector.
  // The transform matrix should be (mat_wo_)^-1^T, which is equal
  // to mat_ow_^-1
  Vector3f out = mat_ow_.MulTranspMatVecMatVec(vec, 0);
  out = Normalize(out);
  return out;
}

Vector3f Transform::O2W_Point(const Vector3f &vec) const {
  return mat_ow_ .MulMatVec(vec, 1);
}

Vector3f Transform::O2W_Vector(const Vector3f &vec) const {
  return mat_ow_ .MulMatVec(vec, 0);
}

Vector3f Transform::O2W_Normal(const Vector3f &vec) const {
  Vector3f out =  mat_wo_.MulTranspMatVecMatVec(vec, 0);
  out = Normalize(out);
  return out;
}

bool Transform::ParseInstruction(
    const TokenizedStatement instruction, 
    const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  if (instruction[0] == "translate") {
    Vector3f t;
    bool good = ParseInstruction_Value<Vector3f>(
      instruction, resource, &t);
    if (good) {
      mat_ow_ = Matrix4x4::Translate(t) * mat_ow_;
      mat_wo_ = mat_wo_ * Matrix4x4::Translate(-t);
    }
    return good;
    
  } else if (instruction[0] == "scale") {
    Float s;
    bool good = ParseInstruction_Value<Float>(
      instruction, resource, &s);
    if (good) {
      mat_ow_ = Matrix4x4::Scale({s, s, s}) * mat_ow_;
      mat_wo_ = mat_wo_ * Matrix4x4::Scale({1/s, 1/s, 1/s});
    }
    return good;
  
  } else if (instruction[0] == "rotate") {
    std::string axis = "y";
    Float angle = 0;
    bool good = ParseInstruction_Pair<std::string, Float>(
      instruction, resource, &axis, &angle);
    if (good) {
      Matrix4x4 (*rot)(const float) = nullptr;
      if (axis == "x") {
        rot = Matrix4x4::RotateX;
      } else if (axis == "y") {
        rot = Matrix4x4::RotateY;
      } else if (axis == "z") {
        rot = Matrix4x4::RotateZ;
      } else {
        std::cerr << "Error: Unkown axis " << axis << "." << std::endl;
        return false;
      }
      mat_ow_ = (*rot)(angle) * mat_ow_;
      mat_wo_ = mat_wo_ * (*rot)(-angle);
    }
    return good;
  
  } else {
    return UnknownInstructionError(instruction);
  }
}

}
