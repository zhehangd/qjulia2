#include "module_sun_light.h"
#include "ui_module_sun_light.h"

#include "core/light/simple.h"

SunLightModule::SunLightModule(QWidget *parent) :
    BaseModule(parent), ui(new Ui::SunLightModule) {
  ui->setupUi(this);
  connect(ui->position, SIGNAL(ValueChanging()), this, SLOT(OnValueChanging()));
  connect(ui->color, SIGNAL(ValueChanging()), this, SLOT(OnValueChanging()));
  connect(ui->position, SIGNAL(ValueChanged()), this, SLOT(OnValueChanged()));
  connect(ui->color, SIGNAL(ValueChanged()), this, SLOT(OnValueChanged()));
}

SunLightModule::~SunLightModule() {
    delete ui;
}

void SunLightModule::AttachEntity(qjulia::Entity *e) {
  entity_ = dynamic_cast<qjulia::SunLight*>(e);
  UpdateWidget();
}

void SunLightModule::UpdateWidget(void) {
  ui->position->SetCoords(-entity_->orientation);
  ui->color->SetRGBColor(entity_->intensity);
}

void SunLightModule::UpdateEntity(void) {
  entity_->orientation = -ui->position->GetCoords();
  entity_->intensity = ui->color->GetRGBColor();
  qDebug() << entity_->intensity[0] << " " << entity_->intensity[1] << " " << entity_->intensity[2] << " ";
}
