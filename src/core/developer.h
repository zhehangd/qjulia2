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

#ifndef QJULIA_DEVELOPER_H_
#define QJULIA_DEVELOPER_H_

#include "base.h"
#include "film.h"
#include "image.h"
#include "vector.h"

namespace qjulia {

/// @brief A Developer process a film into an image
///
/// After an rendering engine uses an integrator to produce a film
/// which contains more or less physically based information of each
/// pixel, a developer is responsible for making an image from the
/// information in the film.
class Developer {
 public:
  CPU_AND_CUDA virtual ~Developer(void) {}
  CPU_AND_CUDA virtual void Develop(const Film &film, Image &dst) = 0;
};

}

#endif
