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

#include "simple.h"
#include "core/resource_mgr.h"

namespace qjulia {

LightRay SunLight::Li(const Point3f &p) const {
  (void)p;
  LightRay lray;
  lray.dist = kInf;
  lray.wi = - Normalize(orientation);
  lray.spectrum = intensity;
  return lray;
}

bool SunLight::ParseInstruction(
    const TokenizedStatement instruction, 
    const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  if (instruction[0] == "intensity") {
    return ParseInstruction_Value<Spectrum>(instruction, resource, &intensity);
  } else if (instruction[0] == "orientation") {
    bool good = ParseInstruction_Value<Vector3f>(
      instruction, resource, &orientation);
    orientation = Normalize(orientation);
    return good;
  } else {
    return UnknownInstructionError(instruction);
  }
}


LightRay PointLight::Li(const Point3f &p) const {
  LightRay lray;
  Vector3f path = (position - p);
  lray.dist = path.Norm();
  lray.wi = path / lray.dist;
  lray.spectrum = intensity / (lray.dist * lray.dist);
  return lray;
}

bool PointLight::ParseInstruction(
    const TokenizedStatement instruction, 
    const ResourceMgr *resource) {
  if (instruction.size() == 0) {return true;}
  if (instruction[0] == "intensity") {
    return ParseInstruction_Value<Spectrum>(instruction, resource, &intensity);
  } else if (instruction[0] == "position") {
    return ParseInstruction_Value<Vector3f>(instruction, resource, &position);
  } else {
    return UnknownInstructionError(instruction);
  }
}


}
