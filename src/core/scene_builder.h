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
#include "qjs_parser.h"
#include "scene.h"

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
  
  /// @brief Returns the pointer to the entity
  ///
  virtual Entity* Get(void) = 0;
  
  virtual Entity* GetDevice(void) = 0;
  
  void SetName(std::string name) {name_ = name;}
  
  std::string GetName(void) const {return name_;}
  
  int GetBaseTypeID(void) const {return btype_id_;}
  
  int GetSpecificTypeID(void) const {return stype_id_;}
    
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
  BT* Get(void) override = 0;
  
  BT* GetDevice(void) override = 0;
};

struct BuildSceneParams {
#ifdef WITH_CUDA
  bool cuda = true;
#else
  bool cuda = false;
#endif
  
  // Name of the world entity for scene building
  // Leave it empty to use the first one found
  std::string world_name = "";
};

/// @brief Creates and maintains the scene data
class SceneBuilder {
 public:
  
   
  /// @brief Parses a QJSContext object
  void ParseSceneDescr(const QJSContext &descr);
  
  /// @brief Parses an EntityDescr object
  void ParseEntityDescr(const QJSBlock &descr);
  
  /// @brief Save to a SceneDescr object
  QJSContext SaveSceneDescr(void);
  
  /// @brief Gets the name of a base type by its ID
  ///
  /// This is a convenience function as the name can be
  /// also obtained by kEntityTypeNames[btype_id].
  std::string GetBaseTypeName(size_t btype_id) const;
  
  /// @brief Gets the name of a specific type by its ID
  std::string GetSpecificTypeName(size_t stype_id) const;
  
  void DebugPrint(void) const;
  
  /// @brief Creates a new entity
  /// Calls with the names of the basic and specific types and
  /// the name assigned to this entity. The name must be unique.
  EntityNode* CreateEntity(std::string btype, std::string stype, std::string name);
  
  /// @brief Creates a new entity. The name must be unique.
  template <typename BT>
  EntityNodeBT<BT>* CreateEntity(std::string stype, std::string name);
  
  /// @brief Searches an entity by its name
  ///
  /// The basic type must be given. Returns nullptr if not found.
  /// If name is empty, returns the first entity matching the basic entity.
  template <typename BT>
  EntityNodeBT<BT>* SearchEntityByName(const std::string &name = {}) const;
  
  EntityNode* SearchEntityByPtr(const Entity *p) const;
  
  std::string SearchEntityNameByPtr(const Entity *p) const;
  
  /// @
  Scene BuildScene(BuildSceneParams params) const;
  
  
  ///@ Get the number of entities
  int NumEntities(void) const {return nodes_.size();}
  
  EntityNode* GetNode(int i) {return nodes_[i].get();}
  
  const EntityNode* GetNode(int i) const {return nodes_[i].get();}
  
  /// @brief Info of a registered type
  struct RegRecord {
    size_t btype_id; // ID of the basic type
    size_t stype_id; // ID of the specific type, same as its table index
    std::string stype_name; // Registered name
    EntityNode*(*fn_create)(void) = nullptr; // function to new such an entity
  };
  
  
  /// @brief Register an entity type
  ///
  /// The basic type class is given as a template argument as its name is
  /// given as a parameter. Specific types of a particular basic type 
  /// must be unique. If a basic type itself is to be registered,
  /// the name should be left empty.
  /// Code that calls this should include \file scene_builder_register.h 
  /// and if CUDA is enabled the file must be compiled with NVCC.
  template <typename ST>
  const RegRecord* Register(std::string stype_name = {});
  
 private:
  
  /// @brief Info of all registered types
  std::vector<RegRecord> reg_table_;
  
  /// @brief Nodes that keep all the created entities
  std::vector<std::unique_ptr<EntityNode> > nodes_;
};

/// @brief Exception for registration failure
struct RegisterFailedExcept : public std::exception {
  RegisterFailedExcept(std::string bname, std::string sname)
    : msg(fmt::format("Cannot register type {}.{}.", bname, sname)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

/// @brief Exception for unknown basic type
struct UnknownBTypeExcept : public std::exception {
  UnknownBTypeExcept(std::string bname)
    : msg(fmt::format("Unknown basic type \"{}\".", bname)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

/// @brief Exception for unknown specific type
struct UnknownSTypeExcept : public std::exception {
  UnknownSTypeExcept(std::string bname, std::string name)
    : msg(fmt::format("Unknown specific type \"{}.{}\".", bname, name)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

/// @brief Exception for unknown entity
struct UnknownEntityExcept : public std::exception {
  UnknownEntityExcept(std::string name)
    : msg(fmt::format("Unknown entity \"{}\".", name)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

/// @brief Exception for unknown entity
struct OccupiedEntityNameExcept : public std::exception {
  OccupiedEntityNameExcept(std::string name)
    : msg(fmt::format("There is already an entity named \"{}\".", name)) {}
  const char* what() const noexcept override {return msg.c_str();}
  std::string msg;
};

template <typename BT>
EntityNodeBT<BT>* SceneBuilder::CreateEntity(std::string stype, std::string name) {
  if (name.empty()) {throw std::runtime_error("Name must not be empty");}
  for (auto &node : nodes_) {
    if (node->GetName() == name) {
      throw OccupiedEntityNameExcept(name);
    }
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
  throw UnknownSTypeExcept(EntityTrait<BT>::name, stype);
}

template <typename BT>
EntityNodeBT<BT>* SceneBuilder::SearchEntityByName(const std::string &name) const {
  for (auto &node : nodes_) {
    if (node->GetName() == name || name.empty()) { // name and btype must both match
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

}

#endif
