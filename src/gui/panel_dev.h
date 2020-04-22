#ifndef QJULIA_DEV_PANEL_H_
#define QJULIA_DEV_PANEL_H_

#include <QMap>
#include <QWidget>

#include "qjulia_context.h"
#include "module_base.h"

namespace Ui {
class DevPanel;
}


struct EntityGUINode {
  
  /// @brief Pointer to the qjulia2 entity
  qjulia::Entity *entity;
  
  /// @brief Pointer to the setting widget
  BaseModule *btype_widget;
  
  BaseModule *stype_widget;
  
  /// @brief Name of the entity
  QString name;
  
  /// @brief Name of the base type of the entity
  QString btype;
  
  /// @brief Name of the specific type of the entity
  QString stype;
};

class DevPanel : public QWidget {
  Q_OBJECT

 public:
  explicit DevPanel(QWidget *parent = 0);
  ~DevPanel();
  
  //void AttachSceneBuilder(qjulia::SceneBuilder *b);
  
  void AttachContext(QJuliaContext *ctx);
  
  void UpdateEntityList(void);
  
  void SwitchToRow(int r);
  
 private:
  
  Ui::DevPanel *ui;
  
  QJuliaContext *ctx_ = nullptr;
  
  // Entity name -> module widget map
  QVector<EntityGUINode> entity_nodes_;
  
  int curr_node_ = -1;
  
  // Temporary flag to make sure AttachSceneBuilder/UpdateEntityList
  // is called only once.
  bool temp_list_is_updated_ = false;
 
signals:
  void ValueChanging(void);
  void ValueChanged(void);
};

#endif // QJULIA_DEV_PANEL_H_
