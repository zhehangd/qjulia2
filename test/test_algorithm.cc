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

#include <gtest/gtest.h> 

#include "qjulia2/core/algorithm.h"

TEST(SolveQuadratic, Q1) {
  qjulia::Float a = 1, b = 5, c = 3;
  qjulia::Float t0 = 0, t1 = 0;
  bool has_roots = qjulia::SolveQuadratic(a, b, c, &t0, &t1);
  EXPECT_TRUE(has_roots);
  EXPECT_NEAR(t0, -4.30278, 0.00001f);
  EXPECT_NEAR(t1, -0.69722, 0.00001f);
}

TEST(SolveQuadratic, Q2) {
  qjulia::Float a = 3, b = 6, c = 3;
  qjulia::Float t0 = -1, t1 = -1;
  bool has_roots = qjulia::SolveQuadratic(a, b, c, &t0, &t1);
  EXPECT_TRUE(has_roots);
  EXPECT_NEAR(t0, -1, 0);
  EXPECT_NEAR(t1, -1, 0);
}

TEST(SolveQuadratic, Q3) {
  qjulia::Float a = 1, b = 2, c = 3;
  qjulia::Float t0 = 66, t1 = 42;
  bool has_roots = qjulia::SolveQuadratic(a, b, c, &t0, &t1);
  EXPECT_FALSE(has_roots);
  EXPECT_EQ(t0, 66);
  EXPECT_EQ(t1, 42);
}
