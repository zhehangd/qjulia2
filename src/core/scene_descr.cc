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

#include "core/scene_descr.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace qjulia {

std::string EntityStatement2Str(const EntityStatement &tokens) {
  std::string str;
  for (const auto &token : tokens) {
    str.append("\"");
    str.append(token);
    str.append("\" ");
  }
  return str;
}

std::string EntityDescr2Str(const EntityDescr &block) {
  std::string str;
  str += fmt::format("Block <type>:{}, <subtype>:{}, <name>:{}\n",
                     block.type, block.subtype, block.name);
  for (const auto &instruction : block.statements) {
    str.append(EntityStatement2Str(instruction));
    str.append("\n");
  }
  return str;
}

namespace {

struct Tokenizer {
 public:
  
  Tokenizer(std::vector<EntityDescr> &dst) : blocks_(dst) {}
   
   
  bool Parse(std::istream &is);
  
  std::string GetErrorMssage(void) const;
  
  void Release(void);

 private:
  
  // Parses a single chracter in a line.
  // Returns true if it expects the next character,
  // false if it expects to skip all remaining chracters in the
  // line (e.g. comments).
  bool ParseChar(char ch);
  
  // Push a token character to the parser.
  
  // Push the next token chracter to the parser.
  void PushTokenChar(char ch);
  
  // Push the next new line chracter to the parser.
  void PushNewline(void);
  
  // Push End-of-File to the parser.
  bool PushEOF(void);
  
  bool BlockBegins(void);
  
  void BlockEnds(void); // TODO Error handling and return error code.
  
  void TokenEnds(void);
  
  void FlushStatementBuf(void);
  
  bool IsSpace(char ch) {return ch == ' ' || ch == '\t';}
  
  // Set if there is error in parsing.
  bool flag_error_ = false;
  
  bool flag_quote_ = false;
  
  // Next encountered character will be escaped when true
  bool flag_escape_ = false;
  
  // Currently inside a token.
  bool flag_inside_token_ = false;
  
  // Currently inside a block.
  bool flag_inside_block_ = false;
  
  bool flag_skip_rest_of_line_ = false;
  
  // Current line number.
  int line_number_ = 0;
  
  // Current line being parsed.
  std::string curr_line_;
  
  const char *error_msg_ = "";
  
  std::string token_buf_;
  EntityStatement statement_buf_;
  EntityDescr block_buf_;
  
  std::vector<EntityDescr> &blocks_;
  
  std::string PrintState(void) {
    std::ostringstream oss;
    oss << flag_escape_ << flag_inside_token_ << flag_inside_block_;
    return oss.str();
  }
};

std::string Tokenizer::GetErrorMssage(void) const {
  return fmt::format("{}\nError at line {}:\n{}", error_msg_, line_number_ + 1, curr_line_);
}

void Tokenizer::PushTokenChar(char ch) {
  if (!flag_inside_token_) {
    flag_inside_token_ = true;
    assert(token_buf_.size() == 0);
  }
  token_buf_.push_back(ch);
}

bool Tokenizer::ParseChar(char ch) {
  //ch = std::tolower(ch);
  if (flag_escape_) {          // Escape
    PushTokenChar(ch);
    flag_escape_ = false;
  } else if (ch == '"') {
    flag_quote_ = !flag_quote_;
  } else if (flag_quote_) {
    PushTokenChar(ch);
  } else if (ch == '#') { // Is comment
    flag_skip_rest_of_line_ = true;
  } else if (IsSpace(ch)) {
    if (flag_inside_token_) {TokenEnds();}
  } else if (ch == '{') {
    if (!BlockBegins()) {return false;}
  } else if (ch == '}') {
    BlockEnds();
  } else if (ch == '\\') {
    flag_escape_ = true;
  } else { // Regular char
    PushTokenChar(ch);
  }
  return true;
}

bool Tokenizer::BlockBegins(void) {
  if (flag_inside_token_) {TokenEnds();}
  const int num_tokens_in_header = statement_buf_.size();
  if (num_tokens_in_header == 1) {
    block_buf_.name = statement_buf_[0];
  } else if (num_tokens_in_header == 2) {
    const auto &type_token = statement_buf_[0];
    auto pos_dot = type_token.find_first_of(".");
    if (pos_dot == std::string::npos) {
      block_buf_.type = type_token;
    } else {
      block_buf_.type = type_token.substr(0, pos_dot);
      block_buf_.subtype = type_token.substr(pos_dot + 1);
    }
    block_buf_.name = statement_buf_[1];
  } else {
    error_msg_ = "Syntax Error: Invalid block header";
    flag_error_ = true;
    return false;
  }
  statement_buf_.clear();
  flag_inside_block_ = true;
  return true;
}

void Tokenizer::BlockEnds(void) {
  if (flag_inside_token_) {TokenEnds();}
  if (statement_buf_.size()) {FlushStatementBuf();}
  blocks_.push_back(block_buf_);
  block_buf_ = EntityDescr();
  flag_inside_block_ = false;
}

void Tokenizer::PushNewline(void) {
  // A newline always terminates token parsing,
  // regardless of escaped or not.
  if (flag_inside_token_) {TokenEnds();}
  // If escaped, the newline is ignored. If not, terminates
  // a statement if it is inside a block.
  if (flag_escape_) {
    flag_escape_ = false;
    return;
  } else {
    if (flag_inside_block_) {FlushStatementBuf();}
  }
  // Increase line number by one.
  // The line string will be overwritten by the next line read,
  // so we do not need to clean it manually.
  ++line_number_;
}

void Tokenizer::TokenEnds(void) {
  if (token_buf_.size() == 0) {return;}
  assert(flag_inside_token_);
  flag_inside_token_ = false;
  statement_buf_.push_back(token_buf_);
  token_buf_.clear();
}

void Tokenizer::FlushStatementBuf(void) {
  if (statement_buf_.size() == 0) {return;}
  assert(!flag_inside_token_); // Must call after token flush.
  assert(flag_inside_block_); // Must be inside a block.
  block_buf_.statements.push_back(statement_buf_);
  statement_buf_.clear();
}

bool Tokenizer::PushEOF(void) {
  assert(token_buf_.size() == 0);
  if (statement_buf_.size() != 0) {
    error_msg_ = "Syntax Error: incomplete content.";
    return false;
  }
  return true;
}

void Tokenizer::Release(void) {
  flag_error_ = false;
  flag_escape_ = false;
  flag_inside_token_ = false;
  flag_inside_block_ = false;
  line_number_ = 0;
  curr_line_.clear();
  token_buf_.clear();
  statement_buf_.clear();
  block_buf_ = EntityDescr();
}

bool Tokenizer::Parse(std::istream &is) {
  Release();
  while (true) {
    std::getline(is, curr_line_);
    if (is.good()) {
      for (char ch : curr_line_) {
        if (!ParseChar(ch)) {return false;}
        if(flag_skip_rest_of_line_) {
          flag_skip_rest_of_line_ = false;
          break;
        }
      }
      PushNewline();
    }
    if (!is.good()) {
      if (!PushEOF()) {break;}
      break;
    }
  }
  return !flag_error_;
}

}

bool TestParamLength(const EntityStatement &statement,
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

SceneDescr LoadSceneFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.good()) {LOG(FATAL) << fmt::format("Cannot open {}.", filename);}
  SceneDescr scene;
  
  Tokenizer tokenizer(scene.entities);
  bool good = tokenizer.Parse(file);
  if (!good) {
    LOG(FATAL) << tokenizer.GetErrorMssage();
    std::cerr << "Encounter error in tokenize the stream." << std::endl;
  }
  return scene;
}


}
