#include "panel_dev.h"
#include "ui_panel_dev.h"

#include <QDebug>
#include <QLabel>

#include "module_camera.h"
#include "module_sun_light.h"
#include "module_placeholder.h"

DevPanel::DevPanel(QWidget *parent)
    : QWidget(parent), ui(new Ui::DevPanel) {
  ui->setupUi(this);
  connect(ui->entityList, &QTableWidget::cellClicked, [this](int r, int) {SwitchToRow(r);});
}

DevPanel::~DevPanel() {
  delete ui;
}

void DevPanel::AttachContext(QJuliaContext *ctx) {
  ctx_= ctx;
  UpdateEntityList();
}

void DevPanel::UpdateEntityList(void) {
  if (temp_list_is_updated_) {
    qDebug() << "Dev: for now AttachSceneBuilder/UpdateEntityList should only be called once";
    temp_list_is_updated_ = true;
  }
  if (ctx_ == nullptr) {
    qDebug() << "Error: No attached scene builder.";
    return;
  }
  auto *build = ctx_->GetSceneBuilder();
  
  int num_entities = build->NumEntities();
  for (int i = 0; i < num_entities; ++i) {
    qjulia::EntityNode *enode = build->GetNode(i);
    qjulia::Entity *entity = enode->Get();
    auto btype_id = enode->GetBaseTypeID();
    auto stype_id = enode->GetSpecificTypeID();
    auto name = QString::fromStdString(enode->GetName());
    auto btype = QString(qjulia::kEntityTypeNames[btype_id]);
    auto stype = QString::fromStdString(build->GetSpecificTypeName(stype_id));
    
    // TODO beware of parenting
    BaseModule *bwidget = ctx_->NewControlWidgetForBaseType(btype_id);
    BaseModule *swidget = ctx_->NewControlWidgetForSpecificType(stype_id);
    if (bwidget == nullptr) {bwidget = new PlaceholderModule();}
    if (swidget == nullptr) {swidget = new PlaceholderModule();}
    bwidget->AttachEntity(entity);
    swidget->AttachEntity(entity);
    
    connect(bwidget, SIGNAL(ValueChanging()), this, SIGNAL(ValueChanging()));
    connect(swidget, SIGNAL(ValueChanging()), this, SIGNAL(ValueChanging()));
    connect(bwidget, SIGNAL(ValueChanged()), this, SIGNAL(ValueChanged()));
    connect(swidget, SIGNAL(ValueChanged()), this, SIGNAL(ValueChanged()));
    
    EntityGUINode gnode {entity, bwidget, swidget, name, btype, stype};
    entity_nodes_.append(gnode);
  }
  
  ui->entityList->setRowCount(entity_nodes_.size());
  for (int i = 0; i < entity_nodes_.size(); ++i) {
    EntityGUINode &node = entity_nodes_[i];
    ui->entityList->setItem(i, 0, new QTableWidgetItem(node.name));
    ui->entityList->setItem(i, 1, new QTableWidgetItem(node.btype));
    ui->entityList->setItem(i, 2, new QTableWidgetItem(node.stype));
  }
}

void DevPanel::SwitchToRow(int i) {
  
  if (curr_node_ >= 0) {
    EntityGUINode &node = entity_nodes_[curr_node_];
    node.btype_widget->hide();
    node.stype_widget->hide();
  }
  
  qDebug()
    << ui->groupBoxBaseLayout->count() << " "
    << ui->groupBoxSpecificLayout->count();
  
  while (auto *item = ui->groupBoxBaseLayout->takeAt(0)) {delete item;}
  while (auto *item = ui->groupBoxSpecificLayout->takeAt(0)) {delete item;}
  
  EntityGUINode &node = entity_nodes_[i];
  node.btype_widget->show();
  node.stype_widget->show();
  curr_node_ = i;
  
  ui->groupBoxBaseLayout->addWidget(node.btype_widget);
  ui->groupBoxSpecificLayout->addWidget(node.stype_widget);
  ui->scrollLayout->update();
  
  qDebug() << node.btype << " " << node.stype << " ";
}
