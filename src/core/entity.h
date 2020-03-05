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

#ifndef QJULIA_ENTITY_H_
#define QJULIA_ENTITY_H_

#include "tokenizer.h"

namespace qjulia {

class ResourceMgr;

class SceneEntity {
 public:
  
  virtual ~SceneEntity(void) {}
  
  /** \breif Returns basic entity type ID
  
  This is used to decide the type of an entity instance.
  This method should be finalized by the base class of each type, such as
  'shape', 'object', and 'material'. Each of these class should also defines
  a static member 'kTypeID' that holds the same id. The ID values are predefined in
  the 'EntityTypeID' enum in 'base.h'.
  */
  virtual EntityType GetType(void) const = 0;
  
  virtual std::string GetImplName(void) const {return "";};
  
  virtual SceneEntity* Clone(void) const = 0;
  
  virtual bool ParseInstruction(const TokenizedStatement instruction, 
                                const ResourceMgr *resource) = 0;
};
/*
class Shape : public SceneEntity {
 public:
  
  std::string GetBaseTypeName(void) const final {return "shape";}
  
  std::string GetSpecificTypeName(void) const = 0;
  
};*/

}

#endif
