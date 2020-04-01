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

#include "core/array2d.h" 
#include "core/vector.h" 

using namespace qjulia;

TEST(Array2D, DefaultConstructor) {
  Array2D<Vector3f> array2d;
  EXPECT_EQ(array2d.Width(), 0);
  EXPECT_EQ(array2d.Height(), 0);
  EXPECT_EQ(array2d.Data(), nullptr);
  EXPECT_FALSE(array2d.HasOwnership());
  EXPECT_EQ(array2d.GetDeleteCount(), 0);
}

TEST(Array2D, NewDataConstructor) {
  Array2D<Vector3f> array2d({48, 36});
  EXPECT_EQ(array2d.Width(), 48);
  EXPECT_EQ(array2d.Height(), 36);
  EXPECT_NE(array2d.Data(), nullptr);
  EXPECT_TRUE(array2d.HasOwnership());
  array2d.Release();
  EXPECT_EQ(array2d.Width(), 0);
  EXPECT_EQ(array2d.Height(), 0);
  EXPECT_EQ(array2d.Data(), nullptr);
  EXPECT_FALSE(array2d.HasOwnership());
  EXPECT_EQ(array2d.GetDeleteCount(), 1);
}

TEST(Array2D, ExistingDataConstructor) {
  Vector3f data[12];
  Array2D<Vector3f> array2d(data, {4, 3});
  EXPECT_EQ(array2d.ArraySize(), Size(4, 3));
  EXPECT_EQ(array2d.Data(), data);
  EXPECT_FALSE(array2d.HasOwnership());
  array2d.Release();
  EXPECT_TRUE(array2d.ArraySize().IsZero());
  EXPECT_EQ(array2d.Data(), nullptr);
  EXPECT_FALSE(array2d.HasOwnership());
  EXPECT_EQ(array2d.GetDeleteCount(), 0);
}

TEST(Array2D, CopyConstructor1) {
  Array2D<Vector3f> a1({64, 48});
  Array2D<Vector3f> a2(a1);
  EXPECT_NE(a1.Data(), nullptr);
  EXPECT_TRUE(a1.HasOwnership());
  EXPECT_EQ(a1.Data(), a2.Data());
  EXPECT_EQ(a1.NumElems(), a2.NumElems());
  EXPECT_FALSE(a2.HasOwnership());
  EXPECT_EQ(a2.GetDeleteCount(), 0);
  a2.Release();
  EXPECT_TRUE(a1.HasOwnership());
  EXPECT_NE(a1.Data(), nullptr);
  EXPECT_FALSE(a2.HasOwnership());
  EXPECT_EQ(a2.Data(), nullptr);
  EXPECT_TRUE(a2.ArraySize().IsZero());
  EXPECT_EQ(a2.GetDeleteCount(), 0);
}

TEST(Array2D, CopyConstructor2) {
  Vector3f data[18];
  Array2D<Vector3f> a1(data, {6, 3});
  Array2D<Vector3f> a2(a1);
  EXPECT_NE(a1.Data(), nullptr);
  EXPECT_FALSE(a1.HasOwnership());
  EXPECT_EQ(a1.Data(), a2.Data());
  EXPECT_EQ(a1.NumElems(), a2.NumElems());
  EXPECT_FALSE(a2.HasOwnership());
  EXPECT_EQ(a2.GetDeleteCount(), 0);
  a2.Release();
  EXPECT_FALSE(a1.HasOwnership());
  EXPECT_NE(a1.Data(), nullptr);
  EXPECT_FALSE(a2.HasOwnership());
  EXPECT_EQ(a2.Data(), nullptr);
  EXPECT_TRUE(a2.ArraySize().IsZero());
  EXPECT_EQ(a2.GetDeleteCount(), 0);
}

TEST(Array2D, MoveConstructor1) {
  Array2D<Vector3f> a1 = Array2D<Vector3f>({6, 3});
  EXPECT_TRUE(a1.HasOwnership());
  EXPECT_NE(a1.Data(), nullptr);
}

TEST(Array2D, MoveConstructor2) {
  Array2D<Vector3f> a1 = Array2D<Vector3f>({6, 3});
  Array2D<Vector3f> a2 = std::move(a1);
  EXPECT_FALSE(a1.HasOwnership());
  EXPECT_TRUE(a2.HasOwnership());
  EXPECT_EQ(a1.Data(), nullptr);
  EXPECT_NE(a2.Data(), nullptr);
  EXPECT_EQ(a1.GetDeleteCount(), 0);
  EXPECT_EQ(a2.GetDeleteCount(), 0);
}

TEST(Array2D, CopyAssignment) {
  Array2D<Vector3f> a1({6, 3});
  Array2D<Vector3f> a2;
  a2 = a1;
  EXPECT_TRUE(a1.HasOwnership());
  EXPECT_FALSE(a2.HasOwnership());
  EXPECT_NE(a1.Data(), nullptr);
  EXPECT_EQ(a1.Data(), a2.Data());
  EXPECT_EQ(a1.GetDeleteCount(), 0);
  EXPECT_EQ(a2.GetDeleteCount(), 0);
}

TEST(Array2D, MoveAssignment1) {
  EXPECT_TRUE(std::is_move_assignable<Array2D<Vector3f>>::value);
  Array2D<Vector3f> a1({6, 3});
  Array2D<Vector3f> a2;
  a2 = std::move(a1);
  EXPECT_FALSE(a1.HasOwnership());
  EXPECT_TRUE(a2.HasOwnership());
  EXPECT_EQ(a1.Data(), nullptr);
  EXPECT_NE(a2.Data(), nullptr);
  EXPECT_EQ(a1.GetDeleteCount(), 0);
  EXPECT_EQ(a2.GetDeleteCount(), 0);
}

TEST(Array2D, MoveAssignment2) {
  EXPECT_TRUE(std::is_move_assignable<Array2D<Vector3f>>::value);
  Array2D<Vector3f> a1({6, 3});
  Array2D<Vector3f> a2({6, 3});
  a2 = std::move(a1);
  EXPECT_FALSE(a1.HasOwnership());
  EXPECT_TRUE(a2.HasOwnership());
  EXPECT_EQ(a1.Data(), nullptr);
  EXPECT_NE(a2.Data(), nullptr);
  EXPECT_EQ(a1.GetDeleteCount(), 0);
  EXPECT_EQ(a2.GetDeleteCount(), 1);
}

TEST(Array2D, CopyToWithResize) {
  Array2D<Vector3f> a1({640, 360});
  Array2D<Vector3f> a2({1920, 1080});
  a1.CopyTo(a2);
  EXPECT_EQ(a1.GetDeleteCount(), 0);
  EXPECT_EQ(a2.GetDeleteCount(), 1);
}

TEST(Array2D, ReturnArray2D1) {
  struct Foo {
    static Array2D<int> Create(void) {Array2D<int> array({3, 4}); return std::move(array);}
  };
  Array2D<int> a1 = Foo::Create();
  EXPECT_TRUE(a1.HasOwnership());
  EXPECT_NE(a1.Data(), nullptr);
  EXPECT_EQ(a1.GetDeleteCount(), 0);
  a1.Release();
  EXPECT_EQ(a1.GetDeleteCount(), 1);
}

TEST(Array2D, ReturnArray2D2) {
  struct Foo {
    static Array2D<int> Create(void) {Array2D<int> array({3, 4}); return std::move(array);}
  };
  Array2D<int> a1({3, 4});
  EXPECT_EQ(a1.GetDeleteCount(), 0);
  a1 = Foo::Create();
  EXPECT_TRUE(a1.HasOwnership());
  EXPECT_NE(a1.Data(), nullptr);
  EXPECT_EQ(a1.GetDeleteCount(), 1);
  a1.Release();
  EXPECT_EQ(a1.GetDeleteCount(), 2);
}
