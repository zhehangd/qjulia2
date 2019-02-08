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

#include "engine.h"

#include <atomic>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "algorithm.h"
#include "camera.h"
#include "efloat.h"
#include "film.h"
#include "light.h"
#include "vector.h"
#include "scene.h"
#include "object.h"
#include "integrator.h"

namespace qjulia {

struct AAFilterSet {
  
  struct Sample {
    Sample(Float x, Float y, Float w) : offset(x, y), w(w) {}
    Vector2f offset;
    Float w = 1;
  };
  
  void Enable(bool enable);
  
  int NumFilters(void) const {return filters.size();}
  
  const Sample& GetFilter(int i) const {return filters[i];}
  
  std::vector<Sample> filters;
};

void AAFilterSet::Enable(bool enable) {
  if (enable) {
    filters = {
      Sample(-0.52f,  0.38f, 0.128f), Sample( 0.41f,  0.56f, 0.119f),
      Sample( 0.27f,  0.08f, 0.294f), Sample(-0.17f, -0.29f, 0.249f),
      Sample( 0.58f, -0.55f, 0.104f), Sample(-0.31f, -0.71f, 0.106f),
    };
  } else {
    filters = {{0, 0, 1}};
  }
}

void RTEngine::Render(const Scene &scene,
                      Integrator &integrator,
                      const Options &option, Film *film) {
  
  int w = option.width;
  int h = option.height;
  assert(w > 0);
  assert(h > 0);
  
  film->Create(w, h);
  
  const Camera *camera = scene.GetActiveCamera();
  
  AAFilterSet antialias;
  antialias.Enable(option.antialias);
  
  std::atomic<int> row_cursor(0);
  
  auto RunThread = [&](void) {
    while(true) {
      int r = row_cursor++;
      if (r >= h) {break;}
      if (r % 200 == 0) {
        std::cerr << "progress: " << r << '/' << h << std::endl;
      }
      auto *row = film->GetRow(r);
      for (int c = 0; c < film->GetWidth(); ++c) {
        auto &pix = row[c];
        Float x, y;
        for (int i = 0; i < antialias.NumFilters(); ++i) {
          const auto &aa = antialias.GetFilter(i);
          Float fr = r + aa.offset[0];
          Float fc = c + aa.offset[1];
          film->GenerateCameraCoords(fr, fc, &x, &y);
          Vector2f p = Vector2f(x, y);
          Ray ray = camera->CastRay(p);
          pix.spectrum += integrator.Li(ray, scene) * aa.w;
        }
      }
    }
  };
  
  int num_threads = num_threads_;
  if (num_threads < 0) {num_threads = std::thread::hardware_concurrency();}
  
  if (num_threads == 0) {
    RunThread();
  } else {
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(RunThread);
    }
    for (auto &thread : threads) {thread.join();}
  }
  
  
}

}
