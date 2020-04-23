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

#ifndef QJULIA_FILM_H_
#define QJULIA_FILM_H_

#include <vector>
#include <array>

#include "spectrum.h"
#include "array2d.h"
#include "integrator_return.h"

namespace qjulia {

class Film : public Array2D<Sample> {
 public:
  CPU_AND_CUDA Film(int w, int h) : Array2D<Sample>({w, h}) {Relocate();}
  CPU_AND_CUDA Film(Size size) : Array2D<Sample>(size) {Relocate();}
  CPU_AND_CUDA Film(Sample *p, int w, int h) : Array2D<Sample>(p, {w, h}) {Relocate();}
  
  CPU_AND_CUDA void Relocate(void);
  CPU_AND_CUDA void Relocate(int x, int y, int w, int h);
  
  CPU_AND_CUDA void GenerateCameraCoords(int i, Float *x, Float *y) const;
  CPU_AND_CUDA void GenerateCameraCoords(Float r, Float c, Float *x, Float *y) const;
  
  CPU_AND_CUDA bool GenerateImageCoords(Float x, Float y, int *i) const;
  CPU_AND_CUDA bool GenerateImageCoords(Float x, Float y, int *r, int *c) const;
  
  CPU_AND_CUDA bool CheckRange(int r, int c) const {return IsValidCoords(r, c);}
  
 private:
  int relocation_x_ = 0;
  int relocation_y_ = 0;
  int relocation_w_ = 0;
  int relocation_h_ = 0;
  int relocation_s_ = 0;
};

//void SaveToPPM(const std::string &filename, const Film &film,
//               Float scale = 255);

}

#endif
