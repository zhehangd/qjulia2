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

#ifndef QJULIA_TRANSFORM_H_
#define QJULIA_TRANSFORM_H_

#include <iostream>
#include <memory>
#include <vector>

#include "entity.h"
#include "vector.h"

namespace qjulia {

class Matrix4x4 : public Vec_<Float, 16> {
 public:
   Matrix4x4(void);
   Matrix4x4(const std::array<Float, 16> vals) {this->vals = vals;}
   
   static Matrix4x4 Identity(void);
   static Matrix4x4 Translate(const Vector3f &T);
   static Matrix4x4 Scale(const Vector3f &S);
   static Matrix4x4 RotateX(const Float angle);
   static Matrix4x4 RotateY(const Float angle);
   static Matrix4x4 RotateZ(const Float angle);
   
   Matrix4x4& operator*=(const Matrix4x4 &mat);
   Matrix4x4 operator*(const Matrix4x4 &mat) const;
   
   // Multiply the matrix by a vector.
   Vector3f MulMatVec(const Vector3f &v, Float w) const;
   
   // Multiply the transformed matrix by a vector.
   Vector3f MulTranspMatVecMatVec(const Vector3f &v, Float w) const;
   
   Float m[4][4] = {};
};

std::ostream& operator<<(std::ostream &os, const Matrix4x4 &mat);
  
class Transform : public SceneEntity {
 public:
  Transform(void) {Identity();}
  Transform(const Vector3f &vec) {(void)vec;}
  
  EntityType GetType(void) const final {return kType;}
  
  void Translate(const Vector3f &translate);
  
  void Scale(const Vector3f &scale);
  
  void Identity(void);
  
  Vector3f W2O_Point(const Vector3f &vec) const;
  
  Vector3f W2O_Vector(const Vector3f &vec) const;
  
  Vector3f W2O_Normal(const Vector3f &vec) const;
  
  Vector3f O2W_Point(const Vector3f &vec) const;
  
  Vector3f O2W_Vector(const Vector3f &vec) const;
  
  Vector3f O2W_Normal(const Vector3f &vec) const;
  
  SceneEntity* Clone(void) const override {return new Transform(*this);}
  
  bool ParseInstruction(const TokenizedStatement instruction, 
                        const ResourceMgr *resource) override;
  
  static const EntityType kType = EntityType::kTransform;
  
  Matrix4x4 mat_ow_; // object to world
  Matrix4x4 mat_wo_; // world to object
};

}

#endif
 
 
