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

#include "core/transform.h"

#include "core/resource_mgr.h"

namespace qjulia {

namespace {
}

std::ostream& operator<<(std::ostream &os, const Matrix4x4 &mat) {
  os << '\n';
  for (int i = 0; i < 4; ++i) {
    os << mat.m[i][0] << ", " << mat.m[i][1] << ", "
       << mat.m[i][2] << ", " << mat.m[i][3] << "\n";
  }
  return os;
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
