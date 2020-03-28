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

#ifndef QJULIA_SCENE_DESCRIPTION_H_
#define QJULIA_SCENE_DESCRIPTION_H_

#include <string>
#include <exception>

#include <fmt/format.h>

#include "base.h"
#include "vector.h"

namespace qjulia {
  
/** \brief Ordered tokens in a line
*/
typedef std::vector<std::string> EntityStatement;

/** \brief Structured tokens representing a block 
  This is a intermediate form, which can be easily generated
  from text without knowing its meaning, and can be parsed
  easily by specific component without knowing text structure.
*/
struct EntityDescr {
  std::string type;
  std::string subtype;
  std::string name;
  std::vector<EntityStatement> statements;
};

struct SceneDescr {
  std::vector<EntityDescr> entities;
};

SceneDescr LoadSceneFile(const std::string &filename);

std::string EntityStatement2Str(const EntityStatement &tokens);

std::string EntityDescr2Str(const EntityDescr &block);


}

#endif
