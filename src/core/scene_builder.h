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

class SceneDescr;
class EntityDescr;

/// @brief Interface of the entity container
///
/// This class has two responsibilities:
/// * Maintains the meta info of an entity such as its name and type.
/// * Maintains the entity's host and device copies.
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
  
  /// @brief ID of the specific type.
  /// This ID is the index of the stype record in the scene builder.
  /// One can 
  size_t stype_id_ = 0;
  size_t btype_id_ = 0;
};

/// @brief EntityNode for a basic type
template<typename BT>
struct EntityNodeBT : public EntityNode {
  virtual BT* Get(void) = 0;
  
  virtual BT* GetDevice(void) = 0;
};

/// @brief Creates and maintains the scene data
class SceneBuilder {
 public:
  
  /// @brief Register an entity type
  ///
  /// The basic type class is given as a template argument as its name is
  /// given as a parameter. Specific types of a particular basic type 
  /// must be unique. If a basic type itself is to be registered,
  /// the name should be left empty.
  template <typename ST>
  bool Register(std::string stype_name = {});
  
  /// @brief Parses a SceneDescr object
  void ParseSceneDescr(const SceneDescr &descr);
  
  /// @brief Parses an EntityDescr object
  void ParseEntityDescr(const EntityDescr &descr);
  
  /// @brief Gets the name of a specific type by its ID
  std::string GetSTypeName(size_t stype_id) const;
  
  void DebugPrint(void) const;
  
  /// @brief Creates a new entity
  /// Calls with the names of the basic and specific types and
  /// the name assigned to this entity. The name must have not
  /// been used for any other of the same basic type.
  EntityNode* CreateEntity(std::string btype, std::string stype, std::string name);
  
  /// @brief Creates a new entity
  template <typename BT>
  EntityNodeBT<BT>* CreateEntity(std::string stype, std::string name);
  
  /// @brief Search an entity by its name
  ///
  /// The basic type must be given. Returns nullptr if not found.
  template <typename BT>
  EntityNodeBT<BT>* SearchEntityByName(const std::string &name);
  
 private:
  
  /// @brief Info of a registered type
  struct RegRecord {
    size_t btype_id; // ID of the basic type
    size_t stype_id; // ID of the specific type, same as its table index
    std::string stype_name; // Registered name
    EntityNode*(*fn_create)(void) = nullptr; // function to new such an entity
  };
  
  /// @brief Info of all registered types
  std::vector<RegRecord> reg_table_;
  
  /// @brief Nodes that keep all the created entities
  std::vector<std::unique_ptr<EntityNode> > nodes_;
};

struct RegisterFailedExcept : public std::exception {
  RegisterFailedExcept(std::string name)
    : msg(fmt::format("Unknown entity \"{}\".", name)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

struct UnknownEntityExcept : public std::exception {
  UnknownEntityExcept(std::string name)
    : msg(fmt::format("Unknown entity \"{}\".", name)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

// We hide the implementation of SceneBuilder::Register inside
// a separate file and include it only when __CUDACC__ is defined.
// By doing so source files that don't call SceneBuilder::Register
// don't have to compiled by NVCC.
#ifdef __CUDACC__
#include "scene_builder.h.in"
#endif

template <typename BT>
EntityNodeBT<BT>* SceneBuilder::CreateEntity(std::string stype, std::string name) {
  if (SearchEntityByName<BT>(name)) {
    LOG(FATAL) << "There is already a " << EntityTrait<BT>::name
              << " entity named " << name << ".";
    return nullptr;
  }
  
  for (const auto &record : reg_table_) {
    if (record.btype_id != EntityTrait<BT>::btype_id) {continue;}
    if (record.stype_name == stype) {
      auto *node = static_cast<EntityNodeBT<BT>*>(record.fn_create());
      node->SetName(name);
      node->stype_id_ = record.stype_id;
      node->btype_id_ = record.btype_id;
      nodes_.emplace_back(node);
      return node;
    }
  }
  LOG(ERROR) << "There is no stype \"" << stype << "\" found for " << EntityTrait<BT>::name << ".";
  return nullptr;
}

template <typename BT>
EntityNodeBT<BT>* SceneBuilder::SearchEntityByName(const std::string &name) {
  for (auto &node : nodes_) {
    if (node->GetName() == name) { // name and btype must both match
      if (EntityTypeID<BT>::val == node->btype_id_) {
        return static_cast<EntityNodeBT<BT>*>(node.get());
      }
    }
  }
  return nullptr;
}

/// @brief Registers all built-in entities
///
/// Typically you should call it immediately after you make a scene builder
/// unless you have other arrangement.
void RegisterDefaultEntities(SceneBuilder &build);



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
