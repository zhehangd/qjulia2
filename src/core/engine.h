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

#ifndef QJULIA_ENGINE_H_
#define QJULIA_ENGINE_H_

#include "spectrum.h"

namespace qjulia {

class Options;
class Scene;
class Film;
class Integrator;
class SceneBuilder;

class Options {
 public:
  bool antialias = true;
  int num_threads = -1;
  bool cuda = true;
};

class RTEngine {
 public:
  
  void Render(SceneBuilder &build, Integrator &integrator,
              const Options &option, Film &film);
  
  float LastRenderTime(void) const {return (float)last_render_time_;}
  
 private:
   
  
  int cu_film_data_size_ = 0;
  Spectrum *cu_film_data_ = nullptr;
  
  float last_render_time_ = 0;
  
  int num_threads_ = -1;
};

}

#endif
