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

namespace qjulia {

template <typename T>
class Array2D {
 public:
  
  using SizeType = int;
  
  Array2D(void) {}
  Array2D(SizeType width, SizeType height, T t = {})
    : width_(width), height_(height), size_(width * height), data_(size_, t) {}
  
  template <typename G>
  static Array2D<T> ZeroLike(const Array2D<G> &src) {return Array2D<T>(src.Width(), src.Height(), {});}
  
  T& At(SizeType r, SizeType c);
  const T& At(SizeType r, SizeType c) const;
  T& At(SizeType i);
  const T& At(SizeType i) const;
  
  T& operator()(SizeType r, SizeType c) {return At(r, c);}
  const T& operator()(SizeType r, SizeType c) const {return At(r, c);}
  T& operator()(SizeType i) {return At(i);}
  const T& operator()(SizeType i) const {return At(i);}
  
  SizeType GetIndex(SizeType r, SizeType c) const {return width_ * r + c;}
  bool IsValidCoords(SizeType r, SizeType c) const;
  
  void Resize(SizeType width, SizeType height);
  
  int Width(void) const {return width_;}
  int Height(void) const {return height_;}
  SizeType Size() const {return size_;}
  
 private:
  SizeType width_ = 0;
  SizeType height_ = 0;
  SizeType size_ = 0;
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
  return GetIndex(r, c) < size_;
}

template <typename T>
void Array2D<T>::Resize(SizeType width, SizeType height) {
  width_ = width;
  height_ = height;
  size_ = width * height;
  data_.resize(size_);
} 

}

#endif
