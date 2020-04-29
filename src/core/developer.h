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
#include "entity.h"

namespace qjulia {

/// @brief A Developer process a film into an image
///
/// After an rendering engine uses an integrator to produce a film
/// which contains more or less physically based information of each
/// pixel, a developer is responsible for making an image from the
/// information in the film.
class Developer : public Entity {
 public:
  
  /// @brief Process a film and accumulate the result in the cache
  virtual void Develop(const Film &film, float w) = 0;
  
  virtual void Init(Size size) = 0;
  
  virtual void Finish(void) = 0;
  
  /// @brief Retrieve cached data from the device
  /// 
  /// This function is called after all the rendering is done,
  /// and the user is going to read the result through the developer.
  /// therefore, only data the user may concern need to be retrieved.
  /// It can be assumed that the given pointer can be cast to the same
  /// type as the class implementing this function in a device kernel.
  virtual void RetrieveFromDevice(Developer *device_ptr) = 0;
  
  virtual void ProduceImage(RGBImage &image) = 0;
};

}

#endif
