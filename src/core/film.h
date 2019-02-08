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

namespace qjulia {

struct FilmSample {
  Spectrum spectrum;
  Float w = 0;
};

class Film {
 public:
  
  Film(void) {}
  Film(int w, int h) {Create(w, h);}
  
  int GetWidth(void) const {return width_;}
  int GetHeight(void) const {return height_;}
  int GetTotal(void) const {return total_;}
  
  void Create(int w, int h);
  void Clean(void);
  
  FilmSample* GetRow(int r) {return buffer_.data() + r * width_;}
  const FilmSample* GetRowConst(int r) const {return buffer_.data() + r * width_;}
  
  FilmSample& At(int r, int c) {return buffer_[GetIndex(r, c)];}
  const FilmSample& At(int r, int c) const {return buffer_[GetIndex(r, c)];}
  
  FilmSample& At(int i) {return buffer_[i];}
  const FilmSample& At(int i) const {return buffer_[i];}
  
  void Relocate(void);
  void Relocate(int x, int y, int w, int h);
  
  int GetIndex(int r, int c) const {return r * height_ + c;}
  
  void GenerateCameraCoords(int i, Float *x, Float *y) const;
  void GenerateCameraCoords(Float r, Float c, Float *x, Float *y) const;
  
  bool GenerateImageCoords(Float x, Float y, int *i) const;
  bool GenerateImageCoords(Float x, Float y, int *r, int *c) const;
  
  bool CheckRange(int r, int c) const;
  
 private:
  typedef std::vector<FilmSample> Buffer;
  
  Buffer buffer_;
  int width_ = 0;
  int height_ = 0;
  int short_ = 0;
  int total_ = 0;
  bool has_relocation_ = false;
  int relocation_x_ = 0;
  int relocation_y_ = 0;
  int relocation_w_ = 0;
  int relocation_h_ = 0;
  int relocation_s_ = 0;
};

void SaveToPPM(const std::string &filename, const Film &film, Float scale = 1);

}

#endif