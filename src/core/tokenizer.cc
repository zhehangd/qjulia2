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

#include "qjulia2/core/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>

namespace qjulia {

std::string TokenizedStatement2Str(const TokenizedStatement &tokens) {
  std::string str;
  for (const auto &token : tokens) {
    str.append("\"");
    str.append(token);
    str.append("\" ");
  }
  return str;
}

std::string TokenizedBlock2Str(const TokenizedBlock &block) {
  std::string str;
  str.append("Block <type>");
  str.append(block.type);
  str.append(" <subtype>");
  str.append(block.subtype);
  str.append(" <name>");
  str.append(block.name);
  str.append("\n");
  for (const auto &instruction : block.instructions) {
    str.append(TokenizedStatement2Str(instruction));
    str.append("\n");
  }
  return str;
}

namespace {

struct Tokenizer {
 public:
  
  bool Parse(std::istream &is, std::vector<TokenizedBlock> *blocks);
  
  void Release(void);

 private:
  
  // Parses a single chracter in a line.
  // Returns true if it expects the next character,
  // false if it expects to skip all remaining chracters in the
  // line (e.g. comments).
  bool ParseChar(char ch);
  
  // Push a token character to the parser.
  
  // Push the next token chracter to the parser.
  void PushRegularChar(char ch);
  
  // Push the next new line chracter to the parser.
  void PushNewline(void);
  
  // Push End-of-File to the parser.
  bool PushEOF(void);
  
  bool BlockBegin(void);
  
  void BlockEnd(void); // TODO Error handling and return error code.
  
  void FlushTokenBuf(void);
  
  void FlushStatementBuf(void);
  
  bool IsSpace(char ch) {return ch == ' ' || ch == '\t';}
  
  // Set if there is error in parsing.
  bool error_ = false;
  
  // Next encountered character will be escaped when true
  bool escape_ = false;
  
  // Currently inside a token.
  bool inside_token_ = false;
  
  // Currently inside a block.
  bool inside_block_ = false;
  
  // Current line number.
  int line_number_ = 0;
  
  // Current line being parsed.
  std::string last_line_;
  
  Token token_buf_;
  TokenizedStatement statement_buf_;
  TokenizedBlock block_buf_;
  
  std::vector<TokenizedBlock> blocks_;
  
  std::string PrintState(void) {
    std::ostringstream oss;
    oss << escape_ << inside_token_ << inside_block_;
    return oss.str();
  }
};

void Tokenizer::PushRegularChar(char ch) {
  if (!inside_token_) {
    inside_token_ = true;
    assert(token_buf_.size() == 0);
  }
  token_buf_.push_back(ch);
}

bool Tokenizer::ParseChar(char ch) {
  ch = std::tolower(ch);
  bool skip = false;
  if (escape_) {          // Escape
    PushRegularChar(ch);
    escape_ = false;
  
  } else if (ch == '#') { // Is comment
    skip = true;
    
  } else if (IsSpace(ch)) {
    if (inside_token_) {FlushTokenBuf();}
    
  } else if (ch == '{') {
    if (!BlockBegin()) {return false;}
    
  } else if (ch == '}') {
    BlockEnd();
  
  } else if (ch == '\\') {
    escape_ = true;
  } else { // Regular char
    PushRegularChar(ch);
  }
  return !skip;
}

bool Tokenizer::BlockBegin(void) {
  if (inside_token_) {FlushTokenBuf();}
  const int header_size = statement_buf_.size();
  if (header_size == 2) {
    block_buf_.type = statement_buf_[0];
    block_buf_.name = statement_buf_[1];
  } else if (header_size == 3) {
    block_buf_.type = statement_buf_[0];
    block_buf_.subtype = statement_buf_[1];
    block_buf_.name = statement_buf_[2];
  } else {
    std::cerr << "Syntax Error: Invalid block header" << std::endl;
    return false;
  }
  statement_buf_.clear();
  inside_block_ = true;
  return true;
}

void Tokenizer::BlockEnd(void) {
  if (inside_token_) {FlushTokenBuf();}
  if (statement_buf_.size()) {FlushStatementBuf();}
  blocks_.push_back(block_buf_);
  block_buf_ = TokenizedBlock();
  inside_block_ = false;
}

void Tokenizer::PushNewline(void) {
  // A newline always terminates token parsing,
  // regardless of escaped or not.
  if (inside_token_) {FlushTokenBuf();}
  // If escaped, the newline is ignored. If not, terminates
  // a statement if it is inside a block.
  if (escape_) {
    escape_ = false;
    return;
  } else {
    if (inside_block_) {FlushStatementBuf();}
  }
  // Increase line number by one.
  // The line string will be overwritten by the next line read,
  // so we do not need to clean it manually.
  ++line_number_;
}

void Tokenizer::FlushTokenBuf(void) {
  if (token_buf_.size() == 0) {return;}
  assert(inside_token_);
  inside_token_ = false;
  statement_buf_.push_back(token_buf_);
  token_buf_.clear();
}

void Tokenizer::FlushStatementBuf(void) {
  if (statement_buf_.size() == 0) {return;}
  assert(!inside_token_); // Must call after token flush.
  assert(inside_block_); // Must be inside a block.
  block_buf_.instructions.push_back(statement_buf_);
  statement_buf_.clear();
}

bool Tokenizer::PushEOF(void) {
  assert(token_buf_.size() == 0);
  if (statement_buf_.size() != 0) {
    std::cerr << "Syntax Error: incomplete content." << std::endl;
    return false;
  }
  return true;
}

void Tokenizer::Release(void) {
  error_ = false;
  escape_ = false;
  inside_token_ = false;
  inside_block_ = false;
  line_number_ = 0;
  last_line_.clear();
  token_buf_.clear();
  statement_buf_.clear();
  block_buf_ = TokenizedBlock();
  blocks_.clear();
}

bool Tokenizer::Parse(
    std::istream &is, std::vector<TokenizedBlock> *blocks) {
  Release();
  while (true) {
    std::getline(is, last_line_);
    
    if (is.good() || last_line_.size() > 0) {
      for (char ch : last_line_) {
        if (!ParseChar(ch)) {break;}
      }
      PushNewline();
    }
    if (!is.good()) {
      PushEOF(); // TODO Error handling
      break;
    }
  }
  blocks_.swap(*blocks);
  return !error_;
  Release();
}

}

bool TestParamLength(const TokenizedStatement &statement,
                     int num_params) {
  assert(statement.size() > 0);
  if (statement.size() == (std::size_t)(num_params + 1)) {
    return true;
  } else {
    std::cerr << "Error: \"" << statement[0] << "\" expecets "
      << num_params << " parameters, got " << (statement.size() - 1)
      << "." << std::endl;
    return false;
  }
}

bool TokenizeSceneFile(
    std::istream &is, std::vector<TokenizedBlock> *blocks) {
  if (!is.good()) {
    std::cerr << "Input stream is not in good state." << std::endl;
    return false;
  }
  Tokenizer tokenizer;
  bool good = tokenizer.Parse(is, blocks);
  if (!good) {
    std::cerr << "Encounter error in tokenize the stream." << std::endl;
  }
  return good;
}


}
