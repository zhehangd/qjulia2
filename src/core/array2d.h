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
#include "stdlib.h"

#include "base.h"


namespace qjulia {

class Size {
 public:
  CPU_AND_CUDA Size(void) {}
  CPU_AND_CUDA Size(SizeType width, SizeType height) : width(width), height(height) {}
  
  CPU_AND_CUDA SizeType Total(void) const {return width * height;}
  
  CPU_AND_CUDA bool IsZero(void) const;
  
  CPU_AND_CUDA bool operator==(const Size &src) const;
  
  SizeType width = 0;
  SizeType height = 0;
};

inline CPU_AND_CUDA bool Size::IsZero(void) const {
  return width == 0 && height == 0;
}

inline CPU_AND_CUDA bool Size::operator==(const Size &src) const {
  return width == src.width && height == src.height;
}

/// @brief 2D Array container
///
/// It is expected to run on CUDA as well, so it cannot use
/// STL containers. We need to implement ourselves. An Array2D
/// may or may not own the data memory. If it is constructed
/// given only a size, or get resized, it will new a block of
/// memory and owns it (meaning it will release it on destruction).
/// If it is assigned from another array, or the data pointer is
/// provided, it will not manage that memory but just pointing to
/// it. So a block of data memory is either owned by only one
/// Array2D or is completely external. This is more like a mix of
/// unique_ptr and shared_ptr.
template <typename T>
class Array2D {
 public:
  
  CPU_AND_CUDA Array2D(void) {}
   
  CPU_AND_CUDA Array2D(Size size);
  
  CPU_AND_CUDA Array2D(T *p, Size size);
  
  /// @brief Make the array points to the same data as src does
  /// but does not hold the ownership.
  ///
  ///
  CPU_AND_CUDA Array2D(const Array2D &src);
  
  CPU_AND_CUDA Array2D(Array2D &&src);
  
  CPU_AND_CUDA ~Array2D(void);
  
  /// @brief Make the array points to the same data as src does
  CPU_AND_CUDA Array2D<T>& operator=(const Array2D &src);
  
  /// @brief Move image
  ///
  /// Points to the same data as src does.
  /// If src owns the data the ownership is moved as well.
  CPU_AND_CUDA Array2D<T>& operator=(Array2D &&src);
  
  void Resize(Size size);
  
  /// @brief Copy the content of an array
  ///
  /// If dst has proper size, data are directly copied to the memory
  /// dst is holding. Otherwise, new memory is allocated for dst.
  void CopyTo(Array2D<T> &dst);
  
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
  
  CPU_AND_CUDA int Width(void) const {return size_.width;}
  CPU_AND_CUDA int Height(void) const {return size_.height;}
  CPU_AND_CUDA Size ArraySize(void) const {return size_;}
  CPU_AND_CUDA SizeType NumElems() const {return size_.Total();}
  CPU_AND_CUDA T* Data(void) {return data_;}
  CPU_AND_CUDA const T* Data(void) const {return data_;}
  
  CPU_AND_CUDA int GetDeleteCount(void) const {return delete_count_;}
  
  CPU_AND_CUDA bool HasOwnership(void) const {return ownership_;}
  
  CPU_AND_CUDA void Release(void);
  
 private:
   
  // If the object owns the data memory
  bool ownership_ = false;
  
  // Size of the array
  Size size_;
  
  // Data memory
  T *data_ = nullptr; // W x H
  
  // This variable records the number of delete calls
  // This should not be overwritten by copying arrays.
  // This is used for debugging and testing
  int delete_count_ = 0;
};

template <typename T>
CPU_AND_CUDA Array2D<T>::Array2D(Size size) {
  if (size.Total() == 0) {
    ownership_ = false;
    size_ = size;
    data_ = nullptr;
  } else {
    ownership_ = true;
    size_ = size;
    data_ = new T[size.Total()]();
  }
}

template <typename T>
CPU_AND_CUDA Array2D<T>::Array2D(T *p, Size size) {
  ownership_ = false;
  size_ = size;
  data_ = p;
}

template <typename T>
CPU_AND_CUDA Array2D<T>::~Array2D(void) {
  Release();
}

template <typename T>
CPU_AND_CUDA Array2D<T>::Array2D(const Array2D &src) {
  ownership_ = false;
  size_ = src.size_;
  data_ = src.data_;
}

template <typename T>
CPU_AND_CUDA Array2D<T>::Array2D(Array2D &&src) {
  ownership_ = src.ownership_;
  src.ownership_ = false;
  data_ = src.data_;
  size_ = src.size_;
  src.Release();
}

template <typename T>
CPU_AND_CUDA void Array2D<T>::Release(void) {
  if (ownership_ && data_) {
    delete[] data_;
    ++delete_count_;
  }
  data_ = nullptr;
  size_ = {};
  ownership_ = false;
}

template <typename T>
void Array2D<T>::Resize(Size size) {
  if (size.width == size_.width && size.height == size_.height) {
    return;
  }
  if (ownership_ && data_) {
    delete[] data_;
    ++delete_count_;
  }
  ownership_ = true;
  size_ = size;
  data_ = new T[size_.Total()]();
}

template <typename T>
void Array2D<T>::CopyTo(Array2D<T> &dst) {
  if (this == &dst) {return;}
  dst.Resize(ArraySize());
  memcpy(dst.Data(), Data(), sizeof(T) * NumElems());
}

template <typename T>
CPU_AND_CUDA Array2D<T>& Array2D<T>::operator=(const Array2D &src) {
  Release();
  ownership_ = false;
  size_ = src.size_;
  data_ = src.data_;
  return *this;
}

template <typename T>
CPU_AND_CUDA Array2D<T>& Array2D<T>::operator=(Array2D &&src) {
  Release();
  ownership_ = src.ownership_;
  src.ownership_ = false;
  data_ = src.data_;
  size_ = src.size_;
  src.Release();
  return *this;
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
