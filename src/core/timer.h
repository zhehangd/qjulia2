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

#ifndef QJULIA_TIMER_H_
#define QJULIA_TIMER_H_

#include <chrono>

#include "base.h"

namespace qjulia {

class Timer {
  public:
  using ClockType = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<ClockType>;
  using TimeDuration = TimePoint::duration;
  using DefualtRepr = Float;
  static constexpr auto Now = ClockType::now;
  
  void Start(void) {srt = Now();}
  
  template <typename T = DefualtRepr>
  T End(void) {
    auto d = Now() - srt;
    acc += d;
    ++count;
    return Value<T>(d);
  }
  
  template <typename T = DefualtRepr>
  T Total(void) const {
    return Value<T>(acc);
  }
  
  template <typename T = DefualtRepr>
  T Average(void) const {
    return Value<T>(acc / count);
  }
  
  template <typename T = DefualtRepr>
  static T Value(TimeDuration v) {
    return std::chrono::duration<T>(v).count();
  }
  
  Timer& operator+=(const Timer &src) {
    acc += src.acc;
    count += src.count;
    return *this;
  }
  
  // Cast any duration to TimeDuration
  template <typename T>
  static TimeDuration MakeDuration(T d) {
    return std::chrono::duration_cast<TimeDuration>(
      std::chrono::duration<T>(d));
  }
  
  TimePoint srt = Now();
  TimeDuration acc = MakeDuration(0);
  int count = 0;
}; 

}

#endif
