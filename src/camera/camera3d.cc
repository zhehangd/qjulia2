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

#include "qjulia2/camera/camera3d.h"
#include "qjulia2/core/resource_mgr.h"

namespace qjulia {

Camera3D::Camera3D(void) {
  position = {0, 0, 1};
  orientation = {0, 0, -1};
  up = {0, 1, 0};
}

void Camera3D::Update(void) {
  orientation = Normalize(orientation);
  up = up - Project(up, orientation);
  up = Normalize(up);
  right = Cross(orientation, up);
}

void Camera3D::LookAt(Vector3f position, Vector3f at, Vector3f up) {
  this->position = position;
  orientation = at - position;
  this->up = up;
  Update();
}

void Camera3D::CenterAround(Float h, Float v, Float radius) {
  h *= kPi / 180.0f;
  v *= kPi / 180.0f;
  Float x = std::cos(h) * std::cos(v) * radius;
  Float z = std::sin(h) * std::cos(v) * radius;
  Float y = std::sin(v) * std::sin(v) * radius;
  LookAt({x, y, z}, {0, 0, 0}, {0, 1, 0});
}

Ray OrthoCamera::CastRay(Point2f pos) const {
  Float x = pos[0];
  Float y = pos[1];
  return Ray(position + right * x + up * y, orientation);
}

bool OrthoCamera::ParseInstruction(
    const TokenizedStatement instruction,  const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  if (instruction[0] == "lookat") {
    Vector3f v3[3];
    bool good = ParseInstruction_Value<Vector3f, 3>(instruction, resource, v3);
    if (good) {LookAt(v3[0], v3[1], v3[2]);}
    return good;
  } else if (instruction[0] == "scale") {
    return ParseInstruction_Value<Float>(instruction, resource, &scale);
  } else {
    return UnknownInstructionError(instruction);
  }
}

PerspectiveCamera::PerspectiveCamera(void) {
  position = {0, 0, 1};
  orientation = {0, 0, -1};
  up = {0, 1, 0};
  focus = 1;
}

Ray PerspectiveCamera::CastRay(Point2f pos) const {
  Float x = pos[0];
  Float y = pos[1];
  Float z = focus;
  Point3f dir = Normalize(orientation * z + up * y + right * x);
  return Ray(position, dir);
}

bool PerspectiveCamera::ParseInstruction(
    const TokenizedStatement instruction,  const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  if (instruction[0] == "lookat") {
    Vector3f v3[3];
    bool good = ParseInstruction_Value<Vector3f, 3>(instruction, resource, v3);
    if (good) {LookAt(v3[0], v3[1], v3[2]);}
    return good;
  } else if (instruction[0] == "focus") {
    return ParseInstruction_Value<Float>(instruction, resource, &focus);
  } else {
    return UnknownInstructionError(instruction);
  }
}

std::string PerspectiveCamera::GetSummary(void) const {
  std::ostringstream oss;
  //oss << "PerspCamera: look from " << position << " toward " << 
  return oss.str();
}

}