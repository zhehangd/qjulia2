#ifndef QJULIA_QJS_PARSER_H_
#define QJULIA_QJS_PARSER_H_

#include <string>
#include <vector>

#include "base.h"

namespace qjulia {
  
struct QJSBlock {
  std::string type;
  std::string subtype;
  std::string name;
  std::vector<Args> statements;
};

struct QJSContext {
  std::vector<QJSBlock> blocks;
};

struct QJSDescription {
  QJSContext scene;
  QJSContext engine;
};

std::string QJSBlock2Str(const QJSBlock &block);

QJSDescription LoadQJSFromStream(std::istream &is);

QJSDescription LoadQJSFromString(const std::string &text);

QJSDescription LoadQJSFromFile(const std::string &filename);

void SaveQJSToStream(std::ostream &os, const QJSDescription &descr);

std::string SaveQJSToString(const QJSDescription &descr);

void SaveQJSToFile(const std::string &filename, const QJSDescription &descr);

}

#endif
