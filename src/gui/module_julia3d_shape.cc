#include "module_julia3d_shape.h"
#include "ui_module_julia3d_shape.h"

#include "core/shape/julia3d.h"

Julia3DShapeModule::Julia3DShapeModule(QWidget *parent) :
    BaseModule(parent), ui(new Ui::Julia3DShapeModule) {
  ui->setupUi(this);
  
  connect(ui->controlBarConst0, SIGNAL(valueChanging(float)), this, SLOT(OnValueChanging()));
  connect(ui->controlBarConst1, SIGNAL(valueChanging(float)), this, SLOT(OnValueChanging()));
  connect(ui->controlBarConst2, SIGNAL(valueChanging(float)), this, SLOT(OnValueChanging()));
  connect(ui->controlBarConst3, SIGNAL(valueChanging(float)), this, SLOT(OnValueChanging()));
  connect(ui->controlBarPrecision, SIGNAL(valueChanging(float)), this, SLOT(OnValueChanging()));
  connect(ui->controlBarUVNormMin, SIGNAL(valueChanging(float)), this, SLOT(OnValueChanging()));
  connect(ui->controlBarUVNormMax, SIGNAL(valueChanging(float)), this, SLOT(OnValueChanging()));
  
  connect(ui->controlBarConst0, SIGNAL(valueChanged(float)), this, SLOT(OnValueChanged()));
  connect(ui->controlBarConst1, SIGNAL(valueChanged(float)), this, SLOT(OnValueChanged()));
  connect(ui->controlBarConst2, SIGNAL(valueChanged(float)), this, SLOT(OnValueChanged()));
  connect(ui->controlBarConst3, SIGNAL(valueChanged(float)), this, SLOT(OnValueChanged()));
  connect(ui->controlBarPrecision, SIGNAL(valueChanged(float)), this, SLOT(OnValueChanged()));
  connect(ui->controlBarUVNormMin, SIGNAL(valueChanged(float)), this, SLOT(OnValueChanged()));
  connect(ui->controlBarUVNormMax, SIGNAL(valueChanged(float)), this, SLOT(OnValueChanged()));
  connect(ui->checkBoxCorssSection, SIGNAL(stateChanged(int)), this, SLOT(OnValueChanged()));
}

Julia3DShapeModule::~Julia3DShapeModule() {
    delete ui;
}

void Julia3DShapeModule::AttachEntity(qjulia::Entity *e) {
  entity_ = dynamic_cast<qjulia::Julia3DShape*>(e);
  UpdateWidget();
}

void Julia3DShapeModule::UpdateWidget(void) {
  qjulia::Quaternion qconst = entity_->GetConstant();
  ui->controlBarConst0->SetValue(qconst[0]);
  ui->controlBarConst1->SetValue(qconst[1]);
  ui->controlBarConst2->SetValue(qconst[2]);
  ui->controlBarConst3->SetValue(qconst[3]);
  ui->controlBarPrecision->SetValue(entity_->GetPrecision());
  ui->controlBarUVNormMin->SetValue(entity_->GetUVBlack());
  ui->controlBarUVNormMax->SetValue(entity_->GetUVWhite());
  ui->checkBoxCorssSection->setChecked(entity_->GetCrossSectionFlag());
}

void Julia3DShapeModule::UpdateEntity(void) {
  qjulia::Quaternion qconst {
    ui->controlBarConst0->GetValue(), ui->controlBarConst1->GetValue(),
    ui->controlBarConst2->GetValue(), ui->controlBarConst3->GetValue(),
  };
  entity_->SetConstant(qconst);
  entity_->SetPrecision(ui->controlBarPrecision->GetValue());
  entity_->SetUVBlack(ui->controlBarUVNormMin->GetValue());
  entity_->SetUVWhite(ui->controlBarUVNormMax->GetValue());
  entity_->SetCrossSectionFlag(ui->checkBoxCorssSection->isChecked());
}
