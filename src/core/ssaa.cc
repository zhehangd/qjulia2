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

#include "ssaa.h"

namespace qjulia {

std::vector<AAFilter> GenerateSSAAFilters(AAOption opt) {
  std::vector<AAFilter> filters;
  if (opt == AAOption::kOff) {
    filters.emplace_back(0, 0, 1);
  } else if (opt == AAOption::kSSAA6x) {
    filters.emplace_back(-0.52f,  0.38f, 0.128f);
    filters.emplace_back( 0.41f,  0.56f, 0.119f);
    filters.emplace_back( 0.27f,  0.08f, 0.294f);
    filters.emplace_back(-0.17f, -0.29f, 0.249f);
    filters.emplace_back( 0.58f, -0.55f, 0.104f);
    filters.emplace_back(-0.31f, -0.71f, 0.106f);
  } else if (opt == AAOption::kSSAA64x) {
    for (int r = 0; r < 8; ++r) {
      for (int c = 0; c < 8; ++c) {
        float fr = r / 8.0f;
        float fc = c / 8.0f;
        filters.emplace_back(fr, fc, 1.0 / 64.0);
      }
    }
  } else if (opt == AAOption::kSSAA256x) {
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 16; ++c) {
        float fr = r / 16.0f;
        float fc = c / 16.0f;
        filters.emplace_back(fr, fc, 1.0 / 256.0);
      }
    }
  }
  return filters;
}

const AAFilter static_aa_samples[6] = {
  AAFilter(-0.52f,  0.38f, 0.128f), AAFilter( 0.41f,  0.56f, 0.119f),
  AAFilter( 0.27f,  0.08f, 0.294f), AAFilter(-0.17f, -0.29f, 0.249f),
  AAFilter( 0.58f, -0.55f, 0.104f), AAFilter(-0.31f, -0.71f, 0.106f),
};

}
