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

#ifndef QJULIA2_SCENE_BUILDER_H_
#define QJULIA2_SCENE_BUILDER_H_

#include "entity.h"

#include <exception>
#include <string>

namespace qjulia {

// Compare if two C strings are the same
inline CPU_AND_CUDA bool EqualString(const char *src1, const char *src2) {
  char c1, c2;
  do {
    c1 = *src1++;
    c2 = *src2++;
  } while (c1 && c2 && c1 == c2);
  return (c1 || c2) == 0x00;
}


struct EntityNode {
  
  virtual ~EntityNode(void) {}
  
  // Returns the pointer to the entity
  // The void* pointer can be cast to the corresponding top-level type.
  virtual Entity* Get(void) = 0;
  
  void SetName(std::string name) {name_ = name;}
  
  std::string GetName(void) const {return name_;}
  
  virtual void AllocateHost(void) = 0;
  
  virtual void ReleaseHost(void) = 0;
  
  virtual void AllocateDevice(void) = 0;
  
  virtual void ReleaseDevice(void) = 0;
  
  virtual void UpdateDevice(void) = 0;
  
  std::string name_;
  size_t stype_id_ = 0;
  size_t btype_id_ = 0;
};

template<typename BT>
struct EntityNodeBT : public EntityNode {
  virtual BT* Get(void) = 0;
  
  virtual BT* GetDevice(void) = 0;
};


// Builds scene data structure from description files and user input.
// It is supposed to run both on GPU and CPU.
class SceneBuilder {
 public:
  
  template <typename ST>
  bool Register(std::string stype_name);
  
  void DebugPrint(void) const;
  
  EntityNodeBT<Camera>* CreateCameraByTypeName(std::string stype, std::string name) {
    return static_cast<EntityNodeBT<Camera>*>(
      CreateEntityByTypeName<Camera>(stype, name));
  }
    
  EntityNodeBT<Light>* CreateLightByTypeName(std::string stype, std::string name) {
    return static_cast<EntityNodeBT<Light>*>(
      CreateEntityByTypeName<Light>(stype, name));
  }
  
  EntityNode* CreateEntity(std::string btype, std::string stype, std::string name);
  
  template <typename BT>
  EntityNodeBT<BT>* CreateEntityByTypeName(std::string stype, std::string name);
  
  template <typename BT>
  EntityNodeBT<BT>* SearchEntityByName(std::string name);
  
  struct EntityRecord {
    size_t btype_id;
    size_t stype_id;
    std::string stype_name;
    EntityNode*(*fn_create)(void) = nullptr;
  };
  
  std::vector<EntityRecord> stype_table_;
  
  // Nodes that hold all the created entities.
  std::vector<std::unique_ptr<EntityNode> > nodes_;
};

// We hide the implementation of SceneBuilder::Register inside
// a separate file and include it only when __CUDACC__ is defined.
// By doing so source files that don't call SceneBuilder::Register
// don't have to compiled by NVCC.
#ifdef __CUDACC__
#include "scene_builder.h.in"
#endif

template <typename BT>
EntityNodeBT<BT>* SceneBuilder::CreateEntityByTypeName(std::string stype, std::string name) {
  if (SearchEntityByName<BT>(name)) {
    LOG(FATAL) << "There is already a " << EntityTypeTraits<BT>::name
              << " entity named " << name << ".";
    return nullptr;
  }
  
  for (const auto &record : stype_table_) {
    if (record.btype_id != EntityTypeTraits<BT>::type_id) {continue;}
    if (record.stype_name == stype) {
      auto *node = static_cast<EntityNodeBT<BT>*>(record.fn_create());
      node->SetName(name);
      node->stype_id_ = record.stype_id;
      node->btype_id_ = record.btype_id;
      nodes_.emplace_back(node);
      return node;
    }
  }
  LOG(ERROR) << "There is no stype \"" << stype << "\" found for " << EntityTypeTraits<BT>::name << ".";
  return nullptr;
}

template <typename BT>
EntityNodeBT<BT>* SceneBuilder::SearchEntityByName(std::string name) {
  for (auto &node : nodes_) {
    if (node->GetName() == name) {
      if (EntityTypeID<BT>::val == node->btype_id_) {
        return static_cast<EntityNodeBT<BT>*>(node.get());
      }
    }
  }
  return nullptr;
}

// Registers all built-in entities
void RegisterDefaultEntities(SceneBuilder &build);

struct UnknownEntityExcept : public std::exception {
  UnknownEntityExcept(std::string name)
    : msg(fmt::format("Unknown entity \"{}\".", name)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

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


}

#endif
