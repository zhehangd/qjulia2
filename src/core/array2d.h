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
  CPU_AND_CUDA Size(void) {}
  CPU_AND_CUDA Size(SizeType width, SizeType height) : width(width), height(height) {}
  CPU_AND_CUDA SizeType Total(void) const {return width * height;}
  SizeType width = 0;
  SizeType height = 0;
};

template <typename T>
class Array2D {
 public:
  
  CPU_AND_CUDA Array2D(Size size);
  
  CPU_AND_CUDA Array2D(Size size, T *p);
  
  CPU_AND_CUDA Array2D(const Array2D &src);
  
  CPU_AND_CUDA ~Array2D(void);
  
  CPU_AND_CUDA T& At(SizeType r, SizeType c);
  CPU_AND_CUDA const T& At(SizeType r, SizeType c) const;
  CPU_AND_CUDA T& At(SizeType i);
  CPU_AND_CUDA const T& At(SizeType i) const;
  
  CPU_AND_CUDA T* Row(SizeType r) {return &At(r, 0);}
  CPU_AND_CUDA const T* Row(SizeType r) const {return &At(r, 0);}
  
  CPU_AND_CUDA T& operator()(SizeType r, SizeType c) {return At(r, c);}
  CPU_AND_CUDA const T& operator()(SizeType r, SizeType c) const {return At(r, c);}
  CPU_AND_CUDA T& operator()(SizeType i) {return At(i);}
  CPU_AND_CUDA const T& operator()(SizeType i) const {return At(i);}
  
  CPU_AND_CUDA SizeType GetIndex(SizeType r, SizeType c) const {return size_.width * r + c;}
  CPU_AND_CUDA bool IsValidCoords(SizeType r, SizeType c) const;
  
  //void Resize(SizeType width, SizeType height);
  
  CPU_AND_CUDA int Width(void) const {return size_.width;}
  CPU_AND_CUDA int Height(void) const {return size_.height;}
  CPU_AND_CUDA Size ArraySize(void) const {return size_;}
  CPU_AND_CUDA SizeType NumElems() const {return size_.Total();}
  CPU_AND_CUDA T* Data(void) {return data_;}
  CPU_AND_CUDA const T* Data(void) const {return data_;}
  
 private:
  bool managed_ = false;
  Size size_;
  T *data_; // W x H
};

template <typename T>
CPU_AND_CUDA Array2D<T>::Array2D(Size size) {
  if (size.Total() == 0) {
    managed_ = false;
    size_ = size;
    data_ = nullptr;
  } else {
    managed_ = true;
    size_ = size;
    data_ = new T[size.Total()]();
  }
}

template <typename T>
CPU_AND_CUDA Array2D<T>::Array2D(Size size, T *p) {
  managed_ = false;
  size_ = size;
  data_ = p;
}

template <typename T>
CPU_AND_CUDA Array2D<T>::~Array2D(void) {
  if (managed_) {delete[] data_;}
}

template <typename T>
CPU_AND_CUDA Array2D<T>::Array2D(const Array2D &src) {
  if (src.managed_) {
    managed_ = true;
    size_ = src.size_;
    data_ = new T[size_.Total()]();
    for (int i = 0; i < size_.Total(); ++i) {data_[i] = src(i);}
  } else {
    managed_ = false;
    size_ = src.size_;
    data_ = src.data_;
  }
}

template <typename T>
CPU_AND_CUDA T& Array2D<T>::At(SizeType r, SizeType c) {
  return At(GetIndex(r, c));
}

template <typename T>
CPU_AND_CUDA const T& Array2D<T>::At(SizeType r, SizeType c) const {
  return At(GetIndex(r, c));
}

template <typename T>
CPU_AND_CUDA T& Array2D<T>::At(SizeType i) {
  return data_[i];
}

template <typename T>
CPU_AND_CUDA const T& Array2D<T>::At(SizeType i) const {
  return data_[i];
}

template <typename T>
CPU_AND_CUDA bool Array2D<T>::IsValidCoords(SizeType r, SizeType c) const {
  return GetIndex(r, c) < size_.Total();
}

}

#endif
