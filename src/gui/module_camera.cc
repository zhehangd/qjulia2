#include "module_camera.h"
#include "ui_module_camera.h"

#include "core/algorithm.h"
#include "core/entity.h"
#include "core/camera.h"

#include <QDebug>

CameraModule::CameraModule(QWidget *parent) :
    BaseModule(parent),
    ui(new Ui::CameraModule) {
  ui->setupUi(this);
  
  connect(ui->target, SIGNAL(ValueChanging()), this, SLOT(OnValueChanging()));
  connect(ui->polPosition, SIGNAL(ValueChanging()), this, SLOT(OnValueChanging()));
  connect(ui->target, SIGNAL(ValueChanged()), this, SLOT(OnValueChanged()));
  connect(ui->polPosition, SIGNAL(ValueChanged()), this, SLOT(OnValueChanged()));
}

CameraModule::~CameraModule() {
  delete ui;
}

void CameraModule::AttachEntity(qjulia::Entity *e) {
  entity_ = dynamic_cast<qjulia::Camera*>(e);
  UpdateWidget();
}

void CameraModule::UpdateWidget(void) {
  auto target = entity_->GetTarget();
  auto cat_rel_position = entity_->GetPosition() - target;
  auto pol_rel_position = qjulia::Cartesian2SphericalCoords(cat_rel_position);
  ui->target->SetCoords(target);
  ui->polPosition->SetSphericalCoords(pol_rel_position);
}

void CameraModule::UpdateEntity(void) {
  auto target = ui->target->GetCoords();
  auto position = ui->polPosition->GetCartesianCoords() + target;
  entity_->LookAt(position, target, {0, 1, 0});
  qDebug() << "Camera Position: " << position[0] << " " << position[1] << " " << position[2];
  qDebug() << "Camera Target: " << target[0] << " " << target[1] << " " << target[2];
}
