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

#ifndef QJULIA_DEVELOPER_SIMPLE_H_
#define QJULIA_DEVELOPER_SIMPLE_H_

#include "core/developer.h"

namespace qjulia {

class SimpleDeveloper : public Developer {
 public:
  
  void Develop(const Film &film, float w) override;
  
  void Init(Size size) override;
  
  void Finish(void) override;
  
  void RetrieveFromDevice(Developer *device_ptr) override;
  
  void ProduceImage(RGBImage &image) override;
  
  struct CachePixel {
    Spectrum spectrum;
    float w = 0;
  };
  
  Array2D<CachePixel> cache_;
};

}

#endif
