#include <algorithm>
#include <cassert>
#include <cctype>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

#include <fmt/format.h>
#include <glog/logging.h>

#include "qjs_parser.h"

namespace qjulia {

namespace {

enum class TokenEndChar {
  kSpace,        //  
  kSemicolon,    // ;
  kLeftBrace,    // {
  kRightBrace,   // }
  kColon,        // :
  kNewline,      // \n
};

class SceneParsingException : public std::runtime_error {
 public:
  SceneParsingException(const std::string &msg) : std::runtime_error(msg) {}
};

/// @brief The text processor at the lowest level
///
/// It splits text into sequences of tokens. Tokens are splitted by whitespaces
/// the sequences are splitted by special chars (newline, semicolon, etc.).
/// Comments are detected and ignored at this level too.
struct Tokenizer {
 public:
  
   /// @brief Read and tokenize a stream
  void Parse(std::istream &is);
  
  /// @brief Bind the function that will be called when a statement is parsed
  ///
  void BindNewStatementFunc(std::function<void(const Args&, TokenEndChar)> fn);
  
  int GetLineNumber(void) {return line_number_;}
  
  void ThrowParsingError(std::string msg);
  
 private:
   
  /// @brief Reset the tokenizer to the initial state
  void Reset(void);
  
  std::function<void(const Args&, TokenEndChar)> fn_callback;
  
  // Parses a single chracter in a line.
  // Returns true if it expects the next character,
  // false if it expects to skip all remaining chracters in the
  // line (e.g. comments).
  void ProcessChar(char ch);
  
  /// @brief Push a regular token char
  ///
  /// The char will either be appended to an unfinished token or
  /// become the head of a new token.
  void PushTokenChar(char ch);
  
  /// @brief Ends the current working token
  ///
  /// This function is called when the parser meets an unescaped special char,
  /// regardless of whether there is an unfinished token.
  void PushTokenEnd(TokenEndChar c);
  
  /// @brief Push the next new line chracter to the parser.
  void ProcessNewline(void);
  
  /// @brief Push End-of-File to the parser.
  bool ProcessEOF(void);
  
  static bool IsSpace(char ch) {return ch == ' ' || ch == '\t';}
  
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
  
  std::vector<std::string> statement_buf_;

  std::string PrintState(void) {
    std::ostringstream oss;
    oss << flag_escape_ << flag_inside_token_ << flag_inside_block_;
    return oss.str();
  }
};

void Tokenizer::BindNewStatementFunc(std::function<void(const Args&, TokenEndChar)> fn) {
  fn_callback = fn;
}

void Tokenizer::ThrowParsingError(std::string msg) {
  throw SceneParsingException(fmt::format(
    "Error at Line {}: {}", line_number_, msg));
}

void Tokenizer::Parse(std::istream &is) {
  Reset();
  while (true) {
    curr_line_.clear();
    std::getline(is, curr_line_);
    for (char ch : curr_line_) {
      ProcessChar(ch);
      if(flag_skip_rest_of_line_) {
        flag_skip_rest_of_line_ = false;
        break;
      }
    }
    if (!is.good()) {
      if (is.eof()) {
        ProcessEOF();
        break;
      } else {
        ThrowParsingError("Stream returns error");
      }
    }
    ProcessNewline();
  }
}

void Tokenizer::ProcessChar(char ch) {
  if (flag_escape_) {          // Escape
    PushTokenChar(ch);
    flag_escape_ = false;
  } else if (ch == '"') {
    flag_quote_ = !flag_quote_;
  } else if (flag_quote_) {
    PushTokenChar(ch);
  } else if (ch == '#') { // Is comment
    flag_skip_rest_of_line_ = true;
    //PushTokenEnd(TokenEndChar::kSharp);
  } else if (IsSpace(ch)) {
    PushTokenEnd(TokenEndChar::kSpace);
  } else if (ch == '{') {
    PushTokenEnd(TokenEndChar::kLeftBrace);
  } else if (ch == '}') {
    PushTokenEnd(TokenEndChar::kRightBrace);
  } else if (ch == ':') {
    PushTokenEnd(TokenEndChar::kColon);
  } else if (ch == ';') {
    PushTokenEnd(TokenEndChar::kSemicolon);
  } else if (ch == '\\') {
    flag_escape_ = true;
  } else { // Regular char
    PushTokenChar(ch);
  }
}

void Tokenizer::ProcessNewline(void) {
  if (flag_quote_) {
    PushTokenChar('\n');
  } else if (flag_escape_) {
    flag_escape_ = false;
    PushTokenEnd(TokenEndChar::kSpace);
  } else {
    PushTokenEnd(TokenEndChar::kNewline);
  }
  ++line_number_;
}

bool Tokenizer::ProcessEOF(void) {
  assert(token_buf_.size() == 0);
  return true;
}

void Tokenizer::PushTokenChar(char ch) {
  if (!flag_inside_token_) {
    flag_inside_token_ = true;
    assert(token_buf_.size() == 0);
  }
  token_buf_.push_back(ch);
}

void Tokenizer::PushTokenEnd(TokenEndChar c) {
  if (token_buf_.size() != 0) {
    assert(flag_inside_token_);
    flag_inside_token_ = false;
    statement_buf_.push_back(token_buf_);
    token_buf_.clear();
  }
  if (c == TokenEndChar::kSpace) {
    // A whitespace does not end the statement, so we keep going
    return;
  } else {
    // End of a statement
    fn_callback(statement_buf_, c);
    statement_buf_.clear();
  }
}

void Tokenizer::Reset(void) {
  flag_escape_ = false;
  flag_inside_token_ = false;
  flag_inside_block_ = false;
  line_number_ = 0;
  curr_line_.clear();
  token_buf_.clear();
}

class BlockParser {
 public:
   
   BlockParser(void);
   
   void Parse(std::istream &is);
   
   void BindNewBlockCallback(std::function<void(const QJSBlock&)> fn);
   
   void BindNewContextCallback(std::function<void(const std::string&)> fn);
   
   void ThrowParsingError(std::string msg) {tokenizer_.ThrowParsingError(msg);}
   
 private:
   void ProcessStatement(const Args &statement, TokenEndChar ec);
   
   std::function<void(const QJSBlock&)> fn_new_block_;
   
   std::function<void(const std::string&)> fn_new_context_;
   
   Tokenizer tokenizer_;
   
   std::vector<QJSBlock> block_sequence_buf_;
   
   QJSBlock block_buf_;
   
   bool flag_inside_block_ = false;
};

BlockParser::BlockParser(void) {
  tokenizer_.BindNewStatementFunc([&](const Args &statement, TokenEndChar ec) {
    ProcessStatement(statement, ec);
  });
}

void BlockParser::BindNewBlockCallback(
    std::function<void(const QJSBlock&)> fn) {
  fn_new_block_ = fn;
}

void BlockParser::BindNewContextCallback(
    std::function<void(const std::string&)> fn) {
  fn_new_context_ = fn;
}

void BlockParser::ProcessStatement(const Args &statement, TokenEndChar ec) {
  assert(ec != TokenEndChar::kSpace);
  
  // A semicolon behave exactly as if it is a newline
  if (ec == TokenEndChar::kSemicolon) {ec = TokenEndChar::kNewline;}
  
  // Ignore empty statements
  if (ec == TokenEndChar::kNewline && statement.empty()) {return;}
  
  if (ec == TokenEndChar::kLeftBrace) { // Start of a block
    if (flag_inside_block_) {
      tokenizer_.ThrowParsingError("Incomplete block");
    }
    int num_header_tokens = statement.size();
    block_buf_ = {};
    if (num_header_tokens == 1) { // <name>
      block_buf_.name = statement[0];
    } else if (num_header_tokens == 2) { // <type>[.subtype] <name>
      const auto &type_token = statement[0];
      auto pos_dot = type_token.find_first_of(".");
      if (pos_dot == std::string::npos) {
        block_buf_.type = type_token;
      } else {
        block_buf_.type = type_token.substr(0, pos_dot);
        block_buf_.subtype = type_token.substr(pos_dot + 1);
      }
      block_buf_.name = statement[1];
    } else {
      tokenizer_.ThrowParsingError("Invalid block header");
    }
    flag_inside_block_ = true;
  } else if (ec == TokenEndChar::kRightBrace) { // End of a block
    if (!flag_inside_block_) {
      tokenizer_.ThrowParsingError("Incomplete block");
    }
    if (!statement.empty()) {
      block_buf_.statements.push_back(statement);
    }
    fn_new_block_(block_buf_);
    block_buf_ = {};
    flag_inside_block_ = false;
  } else if (ec == TokenEndChar::kNewline) { // End of a statement
    if (!flag_inside_block_) {
      tokenizer_.ThrowParsingError("Statement outside a block");
    }
    block_buf_.statements.push_back(statement);
  } else if (ec == TokenEndChar::kColon) { // End of a context
    if (statement.size() != 1) {
      tokenizer_.ThrowParsingError("More than one word found in the context name");
    }
    fn_new_context_(statement[0]);
  }
}

void BlockParser::Parse(std::istream &is) {
  tokenizer_.Parse(is);
}

}

std::string Args2Str(const Args &tokens) {
  std::string str;
  for (const auto &token : tokens) {
    str.append("\"");
    str.append(token);
    str.append("\" ");
  }
  return str;
}

std::string QJSBlock2Str(const QJSBlock &block) {
  std::string str;
  str += fmt::format("Block <type>:{}, <subtype>:{}, <name>:{}\n",
                     block.type, block.subtype, block.name);
  for (const auto &instruction : block.statements) {
    str.append(Args2Str(instruction));
    str.append("\n");
  }
  return str;
}

QJSDescription LoadQJSFromStream(std::istream &is) {
  if (!is.good()) {LOG(FATAL) << fmt::format("Cannot open stream.");}
  
  QJSDescription descr;
  QJSContext *curr_ctx = &descr.scene;
  
  BlockParser scene_parser;
  scene_parser.BindNewBlockCallback([&](const QJSBlock &b) {
    //LOG(INFO) << QJSBlock2Str(b);
    curr_ctx->blocks.push_back(b);
  });
  
  scene_parser.BindNewContextCallback([&](const std::string &name) {
    if (name == "SCENE") {
      curr_ctx = &descr.scene;
    } else if (name == "ENGINE") {
      curr_ctx = &descr.engine;
    } else {
      scene_parser.ThrowParsingError(fmt::format("Undefined context name {}", name));
    }
    //LOG(INFO) << " ===================== Context: " << name << " ===================== ";
  });
  
  scene_parser.Parse(is);
  
  return descr;
}

QJSDescription LoadQJSFromString(const std::string &text) {
  std::istringstream iss(text);
  return LoadQJSFromStream(iss);
}

QJSDescription LoadQJSFromFile(const std::string &filename) {
  std::ifstream file(filename);
  return LoadQJSFromStream(file);
}

void SaveQJSContextToStream(std::ostream &os, const QJSContext &ctx) {
  for (std::size_t k = 0; k < ctx.blocks.size(); ++k) {
    auto &block = ctx.blocks[k];
    CHECK(!block.name.empty());
    if (!block.type.empty()) {
      os << block.type;
      if (!block.subtype.empty()) {
        os << '.' << block.subtype;
      }
      os << ' ' << block.name;
    } else {
      CHECK(block.subtype.empty());
      os << block.name;
    }
    os << " {\n";
    for (const auto s : block.statements) {
      if (!s.size()) {continue;}
      os << "  " << s[0] << " ";
      for (std::size_t i = 1; i < s.size(); ++i) {os << "\"" << s[i] << "\" ";}
      os << '\n';
    }
    os << "}\n\n";
  }
}

void SaveQJSToStream(std::ostream &os, const QJSDescription &descr) {
  os << "SCENE:\n\n";
  SaveQJSContextToStream(os, descr.scene);
  os << '\n';
  os << "ENGINE:\n\n";
  SaveQJSContextToStream(os, descr.engine);
}

std::string SaveQJSToString(const QJSDescription &descr) {
  std::ostringstream oss;
  SaveQJSToStream(oss, descr);
  return oss.str();
}

void SaveQJSToFile(const std::string &filename, const QJSDescription &descr) {
  std::ofstream file(filename);
  SaveQJSToStream(file, descr);
}

}
