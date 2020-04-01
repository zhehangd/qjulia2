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

#include "core/vector.h"

using namespace qjulia;

TEST(BaseVec, Constructors) {
  qjulia::Point2i p1(142, 127);
  EXPECT_EQ(p1[0], 142);
  EXPECT_EQ(p1[1], 127);
  
  qjulia::Point3f p2(242.f, 227.1f, 233.f);
  EXPECT_EQ(p2[0], 242.f);
  EXPECT_EQ(p2[1], 227.1f);
  EXPECT_EQ(p2[2], 233.f);
  
  qjulia::Point3f p3(342.1f, -327.f);
  EXPECT_EQ(p3[0], 342.1f);
  EXPECT_EQ(p3[1], -327.f);
  EXPECT_EQ(p3[2], 0.f);
  EXPECT_NE(p3[0], 342.0f);
  
  qjulia::Quaternion p4(-442.f, -427.f, 4.99f, 1.12f);
  EXPECT_EQ(p4[0], -442.f);
  EXPECT_EQ(p4[1], -427.f);
  EXPECT_EQ(p4[2], 4.99f);
  EXPECT_EQ(p4[3], 1.12f);
  
  qjulia::Quaternion p5(-542.f, -527.f, 5.5f, -5.f);
  EXPECT_EQ(p5[0], -542.f);
  EXPECT_EQ(p5[1], -527.f);
  EXPECT_EQ(p5[2], 5.5f);
  EXPECT_EQ(p5[3], -5.f);
}

TEST(BaseVec, MemorySize) {
  EXPECT_EQ(sizeof(Vector3f), 3 * sizeof(Float));
}

TEST(BaseVec, Equality) {
  qjulia::Point3f p1(7.5f, - 8.1f);
  qjulia::Point3f p2(7.5f, - 8.1f);
  qjulia::Point3f p3(7.5f, - 8.0f);
  EXPECT_EQ(p1, p2);
  EXPECT_NE(p1, p3);
}

TEST(BaseVec, AdditionAndSubtraction) {
  qjulia::Point2i p1(14, 42);
  qjulia::Point2i p2(-5, 277);
  qjulia::Point2i p;
  int c = -2;
  
  p = p1;
  p += p2;
  EXPECT_EQ(p[0], 9);
  EXPECT_EQ(p[1], 319);
  
  p = p1;
  p -= p2;
  EXPECT_EQ(p[0], 19);
  EXPECT_EQ(p[1], -235);
  
  p = p1 + p2;
  EXPECT_EQ(p[0], 9);
  EXPECT_EQ(p[1], 319);
  
  p = p2 - p1;
  EXPECT_EQ(p[0], -19);
  EXPECT_EQ(p[1], 235);
  
  p = p1;
  p += c;
  EXPECT_EQ(p[0], 12);
  EXPECT_EQ(p[1], 40);
  
  p = p1;
  p -= c;
  EXPECT_EQ(p[0], 16);
  EXPECT_EQ(p[1], 44);
  
  p = p1 + c;
  EXPECT_EQ(p[0], 12);
  EXPECT_EQ(p[1], 40);
  
  p = p1 - c;
  EXPECT_EQ(p[0], 16);
  EXPECT_EQ(p[1], 44);
}


TEST(BaseVec, MultiplicationAndDivision) {
  qjulia::Point2i p1(14, 42);
  qjulia::Point2i p;
  int c = -2;
  
  p = p1;
  p *= c;
  EXPECT_EQ(p[0], -28);
  EXPECT_EQ(p[1], -84);
  
  p = p1;
  p /= c;
  EXPECT_EQ(p[0], -7);
  EXPECT_EQ(p[1], -21);
  
  p = p1 * c;
  EXPECT_EQ(p[0], -28);
  EXPECT_EQ(p[1], -84);
  
  p = p1 / c;
  EXPECT_EQ(p[0], -7);
  EXPECT_EQ(p[1], -21);
}

TEST(BaseVec, Distance) {
  qjulia::Point3f p1(3.2f, 4.8f);
  qjulia::Point3f p2(-7.7f, 1.9f);
  EXPECT_NEAR(qjulia::Dist(p1, p2), 11.27918f, 0.00001f);
}

TEST(Point, Dot) {
  qjulia::Point3f p1(3.2f, 4.8f);
  qjulia::Point3f p2(-7.7f, 1.9f);
  EXPECT_NEAR(qjulia::Dot(p1, p2), -15.52f, 0.001f);
}

TEST(Point, Project) {
  qjulia::Point3f a(3.2f, 4.8f, 0.0f);
  qjulia::Point3f b(-7.7f, 1.9f, 2.3f);
  qjulia::Point3f p1 = qjulia::Project(a, b);
  qjulia::Point3f p2 = qjulia::Project(b, a);
  
  EXPECT_NEAR(p1[0], 1.7525, 0.0001f);
  EXPECT_NEAR(p1[1], -0.4324, 0.0001f);
  EXPECT_NEAR(p1[2], -0.5235, 0.0001f);
  EXPECT_NEAR(p2[0], -1.4923, 0.0001f);
  EXPECT_NEAR(p2[1], -2.2385, 0.0001f);
  EXPECT_NEAR(p2[2], 0, 0.0001f);
}

TEST(Quaternion, Multiplication) {
  qjulia::Quaternion p1(3.2f, 4.8f, -7.7f, 1.9f);
  qjulia::Quaternion p2(-0.5f, 1.2f, -3.3f, 4.2f);
  
  qjulia::Quaternion p = p1;
  p *= p2;
  EXPECT_NEAR(p.Real(), -40.75, 0.005f);
  EXPECT_NEAR(p.ImagI(), -24.63, 0.005f);
  EXPECT_NEAR(p.ImagJ(), -24.59, 0.005f);
  EXPECT_NEAR(p.ImagK(), 5.89, 0.005f);
}
