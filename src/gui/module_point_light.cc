#include "module_point_light.h"
#include "ui_module_point_light.h"

#include "core/light/simple.h"

PointLightModule::PointLightModule(QWidget *parent) :
    BaseModule(parent), ui(new Ui::PointLightModule) {
  ui->setupUi(this);
  connect(ui->position, SIGNAL(ValueChanging()), this, SLOT(OnValueChanging()));
  connect(ui->color, SIGNAL(ValueChanging()), this, SLOT(OnValueChanging()));
  connect(ui->position, SIGNAL(ValueChanged()), this, SLOT(OnValueChanged()));
  connect(ui->color, SIGNAL(ValueChanged()), this, SLOT(OnValueChanged()));
}

PointLightModule::~PointLightModule() {
    delete ui;
}

void PointLightModule::AttachEntity(qjulia::Entity *e) {
  entity_ = dynamic_cast<qjulia::PointLight*>(e);
  UpdateWidget();
}

void PointLightModule::UpdateWidget(void) {
  ui->position->SetCoords(entity_->position);
  ui->color->SetRGBColor(entity_->intensity);
}

void PointLightModule::UpdateEntity(void) {
  entity_->position = ui->position->GetCoords();
  entity_->intensity = ui->color->GetRGBColor();
  qDebug() << "PointLight Intensity: " << entity_->intensity[0] << " " << entity_->intensity[1] << " " << entity_->intensity[2];
  qDebug() << "PointLight Position: " << entity_->position[0] << " " << entity_->position[1] << " " << entity_->position[2];
}
