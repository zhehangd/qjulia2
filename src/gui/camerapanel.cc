#include "camerapanel.h"
#include "ui_camerapanel.h"

#include <QDebug>

CameraPanel::CameraPanel(QWidget *parent)
    : Panel(parent), ui(new Ui::CameraPanel) {
  ui->setupUi(this);
  
  connect(ui->controlBarCamTargetX, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarCamTargetY, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarCamTargetZ, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarCamAzi, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarCamAlt, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarCamDist, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarFov, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarLightLumin, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  
  connect(ui->controlBarCamTargetX, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarCamTargetY, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarCamTargetZ, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarCamAzi, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarCamAlt, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarCamDist, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarFov, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarLightLumin, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
}

CameraPanel::~CameraPanel() {
  delete ui;
}

void CameraPanel::LinkToOptions(SceneCtrlParams *opts) {
  opts_ = opts;
  ui->controlBarCamTargetX->SetValue(opts_->camera_target[0]);
  ui->controlBarCamTargetY->SetValue(opts_->camera_target[1]);
  ui->controlBarCamTargetZ->SetValue(opts_->camera_target[2]);
  ui->controlBarCamAzi->SetValue(opts_->camera_pose[0]);
  ui->controlBarCamAlt->SetValue(opts_->camera_pose[1]);
  ui->controlBarCamDist->SetValue(opts_->camera_pose[2]);
  ui->controlBarFov->SetValue(opts_->camera_fov);
  ui->controlBarLightLumin->SetValue(opts_->camera_headlight_lumin);
}

void CameraPanel::CollectParams(void) {
  if (opts_) {
    opts_->camera_target = {
      ui->controlBarCamTargetX->GetValue(),
      ui->controlBarCamTargetY->GetValue(),
      ui->controlBarCamTargetZ->GetValue(),
    };
    opts_->camera_pose = {
      ui->controlBarCamAzi->GetValue(),
      ui->controlBarCamAlt->GetValue(),
      ui->controlBarCamDist->GetValue(),
    };
    opts_->camera_fov = ui->controlBarFov->GetValue();
    opts_->camera_headlight_lumin = ui->controlBarLightLumin->GetValue();
  }
}

