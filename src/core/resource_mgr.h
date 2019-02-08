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

#ifndef QJULIA_RESOURCE_MANAGER_H_
#define QJULIA_RESOURCE_MANAGER_H_

#include <iostream>
#include <functional>
#include <vector>
#include <memory>
#include <map>
#include <type_traits>

#include "base.h"
#include "entity.h"
#include "tokenizer.h"

namespace qjulia {

// IDs:
// Object 0
// Shape 1
// Transform 2
// Material 3
// Light 4
// Camera 5
// Scene 6

typedef std::map<std::string, std::unique_ptr<SceneEntity> > EntityMap;
typedef std::array<EntityMap, kNumEntityTypes> EntityMapArray;

class Scene;

class ResourceMgr {
 public:
  
  /** \brief Register a template entity for scene parsing
  By calling this function the manager is informed of a specific 
  entity type.
  */
  bool RegisterPrototypeEntity(SceneEntity *entity);
  
  void PrintPrototypes(std::ostream &os) const;
  
  void PrintEntities(std::ostream &os) const;
  
  bool LoadSceneDescription(const std::string &filename);
  
  bool ParseFromTokenizedBlock(const TokenizedBlock &block);
  
  Scene* GetScene(void) const;
  
  //Scene* GetScene(const std::string &name);
    
  template <typename T>
  T* GetEntity(std::string name) const;
  
  // Array[ClassID][EntityName]
  EntityMapArray entity_maps_;
  
  // Prototype design pattern
  // http://www.gameprogrammingpatterns.com/prototype.html
  // Array[ClassID][SubtypeName]
  EntityMapArray prototype_entities_;
  
  
 private:
   
   template <typename T>
   bool TryParseWithType(const TokenizedBlock &block);
};

template <typename T>
T* ResourceMgr::GetEntity(std::string name) const {
  const int eid = GetEntityTypeID(T::kType);
  auto &entity_map = entity_maps_[eid];
  auto it = entity_map.find(name);
  if (it != entity_map.end()) {
    return dynamic_cast<T*>(it->second.get());
  } else {
    return nullptr;
  }
}

/** \brief Print error messsage for unknown instruction name
*/
inline bool UnknownInstructionError(const TokenizedStatement &instruction) {
  const int num_args = instruction.size();
  assert(num_args > 0);
  (void)num_args;
  std::cerr << "Error: Unknown instruction name \""
    << instruction[0] << "\"." << std::endl;
  return false;
}

template <typename T, int C = 1>
bool ParseInstruction_Pointer(
    const TokenizedStatement &instruction,  const ResourceMgr *resource,
    const T **dst) {
  const int num_args = instruction.size() - 1;
  assert(num_args >= 0); // In the case the func should not be called.
  
  if (num_args != C) {
    std::cerr << "Expected " << C << " argement(s), got "
              << num_args << "." << std::endl;
    return false;
  }
  for (int i = 0; i < C; ++i) {
    const std::string &name = instruction[1 + i];
    T *entity = resource->GetEntity<T>(name);
    if (entity == nullptr) {
      std::cerr << "Cannot find the entity \"" << name << "\"." << std::endl;
      return false;
    }
    dst[i] = entity;
  }
  return true;
}

template <typename T, int C = 1>
bool ParseInstruction_Value(
    const TokenizedStatement &instruction,  const ResourceMgr *resource,
    T *dst) {
  (void)resource;
  const int num_args = instruction.size() - 1;
  assert(num_args >= 0); // In the case the func should not be called.
  
  if (num_args != C) {
    std::cerr << "Expected " << C << " argement(s), got "
              << num_args << "." << std::endl;
    return false;
  }
  bool good = true;
  for (int i = 0; i < num_args; ++i) {
    good &= ParseToken(instruction[1 + i], &(dst[i]));
    if (!good) {break;}
  }
  return good;
}

template <typename T1, typename T2>
bool ParseInstruction_Pair(
    const TokenizedStatement &instruction,  const ResourceMgr *resource,
    T1 *dst1, T2 *dst2) {
  (void)resource;
  const int num_args = instruction.size() - 1;
  assert(num_args >= 0); // In the case the func should not be called.
  
  if (num_args != 2) {
    std::cerr << "Expected 2 argement(s), got "
              << num_args << "." << std::endl;
    return false;
  }
  bool good = true;
  if (good) {good &= ParseToken(instruction[1], dst1);}
  if (good) {good &= ParseToken(instruction[2], dst2);}
  return good;
}

}

#endif
