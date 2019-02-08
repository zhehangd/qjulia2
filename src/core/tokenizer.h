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

#ifndef QJULIA_TOKENIZER_H_
#define QJULIA_TOKENIZER_H_

#include <vector>
#include <memory>

#include <glog/logging.h>

//#include "vector.h"
#include "base.h"
#include "vector.h"
#include "messages.h"

namespace qjulia {

/** \brief An element in a scene descrition
A token is any string containing only characters with graphical
representation. That is, common symbols except for whitespaces.
*/
typedef std::string Token;

/** \brief Ordered tokens in a line
*/
typedef std::vector<Token> TokenizedStatement;

/** \brief A sequence of token lines
*/
typedef std::vector<TokenizedStatement> TokenizedStatementSeq;

/** \brief Structured tokens representing a block 
  This is a intermediate form, which can be easily generated
  from text without knowing its meaning, and can be parsed
  easily by specific component without knowing text structure.
*/
struct TokenizedBlock {
  Token type;
  Token subtype;
  Token name;
  TokenizedStatementSeq instructions;
};

bool TokenizeSceneFile(
  std::istream &is, std::vector<TokenizedBlock> *blocks);

std::string TokenizedStatement2Str(const TokenizedStatement &tokens);

std::string TokenizedBlock2Str(const TokenizedBlock &block);

// Tools that help token parsing
// =====================================================

template <typename T>
bool ParseToken(const std::string &token, T *val) {
  std::istringstream iss(token);
  if (!iss.good()) {
    BadStreamMessage();
    return false;
  }
  iss >> (*val);
  if (!iss.fail()) {
    return true;
  } else {
    CannotParseTokenMessage(token);
    return false;
  }
}

template <>
inline bool ParseToken<bool>(const std::string &token, bool *val) {
  std::istringstream iss(token);
  std::string text;
  iss >> text;
  if (iss.good()) {
    BadStreamMessage();
    return false;
  }
  if (text == "on" || text == "true") {
    *val = true;
    return true;
  } else if (text == "off" || text == "false") {
    *val = false;
    return true;
  } else {
    CannotParseTokenMessage(token);
    return false;
  }
}

bool TestParamLength(const TokenizedStatement &statement,
                     int num_params);


}

#endif
 
