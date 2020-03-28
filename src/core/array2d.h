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

#ifndef QJULIA_ARRAY2D_H_
#define QJULIA_ARRAY2D_H_

#include <vector>

#include "base.h"

namespace qjulia {

class Size {
 public:
  Size(void) {}
  Size(SizeType width, SizeType height) : width(width), height(height) {}
  SizeType Total(void) const {return width * height;}
  SizeType width = 0;
  SizeType height = 0;
};
  

template <typename T>
class Array2D {
 public:
  
  Array2D(Size size, T t = {}) : size_(size), data_(size_.Total(), t) {}
  Array2D(SizeType width, SizeType height, T t = {})
    : size_(width, height), data_(size_.Total(), t) {}
  
  template <typename G>
  static Array2D<T> ZeroLike(const Array2D<G> &src) {return Array2D<T>(src.Width(), src.Height(), {});}
  
  T& At(SizeType r, SizeType c);
  const T& At(SizeType r, SizeType c) const;
  T& At(SizeType i);
  const T& At(SizeType i) const;
  
  T* Row(SizeType r) {return &At(r, 0);}
  const T* Row(SizeType r) const {return &At(r, 0);}
  
  T& operator()(SizeType r, SizeType c) {return At(r, c);}
  const T& operator()(SizeType r, SizeType c) const {return At(r, c);}
  T& operator()(SizeType i) {return At(i);}
  const T& operator()(SizeType i) const {return At(i);}
  
  SizeType GetIndex(SizeType r, SizeType c) const {return size_.width * r + c;}
  bool IsValidCoords(SizeType r, SizeType c) const;
  
  //void Resize(SizeType width, SizeType height);
  
  int Width(void) const {return size_.width;}
  int Height(void) const {return size_.height;}
  Size ArraySize(void) const {return size_;}
  SizeType NumElems() const {return size_.Total();}
  T* Data(void) {return data_.data();}
  const T* Data(void) const {return data_.data();}
  
 private:
  Size size_;
  std::vector<T> data_; // W x H
};

template <typename T>
T& Array2D<T>::At(SizeType r, SizeType c) {
  return At(GetIndex(r, c));
}

template <typename T>
const T& Array2D<T>::At(SizeType r, SizeType c) const {
  return At(GetIndex(r, c));
}

template <typename T>
T& Array2D<T>::At(SizeType i) {
  return data_[i];
}

template <typename T>
const T& Array2D<T>::At(SizeType i) const {
  return data_[i];
}

template <typename T>
bool Array2D<T>::IsValidCoords(SizeType r, SizeType c) const {
  return GetIndex(r, c) < size_.Total();
}

}

#endif
