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

#ifndef QJULIA_ARG_PARSE_H_
#define QJULIA_ARG_PARSE_H_

#include "scene_builder.h"

#include <exception>
#include <string>

namespace qjulia {

// Provides two functions with many overloads
// ParseArg and ParseEntityNode
  
struct ParseFailure : public std::exception {
  ParseFailure(std::string arg, std::string type)
    : msg(fmt::format("Cannot interpret \"{}\" as {}.", arg, type)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

struct UnknownCommand : public std::exception {
  UnknownCommand(std::string cmd)
    : msg(fmt::format("Unknown command \"{}\".", cmd)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

inline void ParseArg(std::string arg, Float &dst) {
  try {
#ifdef DOUBLE_PRECISION
    dst = std::stod(arg);
#else
    dst = std::stof(arg);
#endif
  }
  catch (const std::invalid_argument& ia) {
    throw ParseFailure(arg, "float");
  }
}

inline void ParseArg(std::string arg, int &dst) {
  try {
    dst = std::stoi(arg);
  }
  catch (const std::invalid_argument& ia) {
    throw ParseFailure(arg, "int");
  }
}

inline void ParseArg(std::string arg, bool &dst) {
  if (arg == "on" || arg == "true" || arg == "1") {
    dst = true;
  } else if (arg == "off" || arg == "false" || arg == "0") {
    dst = false;
  } else {
    throw ParseFailure(arg, "bool");
  }
}

template <int C>
void ParseArg(std::string arg, Vec_<Float, C> &dst) {
  try {
    std::size_t pos = 0;
    for (int i = 0; i < C; ++i) {
      std::size_t idx = 0;
      dst[i] = std::stof(arg.substr(pos), &idx);
      pos += idx + 1;
    }
  }
  catch (const std::invalid_argument& ia) {
    throw ParseFailure(arg, "Vector" + std::to_string(C) + "f");
  }
}

template <int C>
void ParseArg(std::string arg, Vec_<int, C> &dst) {
  try {
    std::size_t pos = 0;
    for (int i = 0; i < C; ++i) {
      std::size_t idx = 0;
      dst[i] = std::stoi(arg.substr(pos), &idx);
      pos += idx + 1;
    }
  }
  catch (const std::invalid_argument& ia) {
    throw ParseFailure(arg, "Vector" + std::to_string(C) + "i");
  }
}

template <typename BT>
EntityNodeBT<BT>* ParseEntityNode(const std::string &name, SceneBuilder *build) {
  auto *node = build->SearchEntityByName<BT>(name);
  if (!node) {throw UnknownEntityExcept(name);}
  return node;
}

// TODO remove
template <typename BT>
BT* ParseEntity(const std::string &name, SceneBuilder *build) {
  return ParseEntityNode<BT>(name, build)->Get();
}

inline std::string ToString(int v) {
  return fmt::format("{}", v);
}

inline std::string ToString(Float v) {
  return fmt::format("{}", v);
}

template <typename T, int C>
std::string ToString(Vec_<T, C> v) {
  std::vector<T> vec(v.vals, v.vals + C);
  return fmt::format("{}", fmt::join(vec, ","));
}

}

#endif
