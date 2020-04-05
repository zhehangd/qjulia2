#include "fractalpanel.h"
#include "ui_fractalpanel.h"

#include <QDebug>

FractalPanel::FractalPanel(QWidget *parent)
    : Panel(parent), ui(new Ui::FractalPanel) {
  ui->setupUi(this);
  
  connect(ui->controlBarConst0, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarConst1, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarConst2, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarConst3, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarPrecision, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarUVNormMin, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  connect(ui->controlBarUVNormMax, SIGNAL(valueChanging(float)), this, SLOT(onRealtimeParamsChanging(float)));
  
  connect(ui->controlBarConst0, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarConst1, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarConst2, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarConst3, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarPrecision, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarUVNormMin, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->controlBarUVNormMax, SIGNAL(valueChanged(float)), this, SLOT(onRealtimeParamsChanged(float)));
  connect(ui->checkBoxCorssSection, SIGNAL(stateChanged(int)), this, SLOT(onRealtimeParamsChanged(int)));
}

FractalPanel::~FractalPanel() {
  delete ui;
}

void FractalPanel::LinkToOptions(SceneCtrlParams *opts) {
  opts_ = opts;
  ui->controlBarConst0->SetValue(opts_->fractal_constant[0]);
  ui->controlBarConst1->SetValue(opts_->fractal_constant[1]);
  ui->controlBarConst2->SetValue(opts_->fractal_constant[2]);
  ui->controlBarConst3->SetValue(opts_->fractal_constant[3]);
  ui->controlBarPrecision->SetValue(opts_->fractal_precision);
  ui->controlBarUVNormMin->SetValue(opts_->fractal_uv_black);
  ui->controlBarUVNormMax->SetValue(opts_->fractal_uv_white);
  ui->checkBoxCorssSection->setChecked(opts_->fractal_cross_section);
}

void FractalPanel::CollectParams(void) {
  if (opts_) {
    opts_->fractal_constant = {
      ui->controlBarConst0->GetValue(),
      ui->controlBarConst1->GetValue(),
      ui->controlBarConst2->GetValue(),
      ui->controlBarConst3->GetValue(),
    };
    opts_->fractal_precision = ui->controlBarPrecision->GetValue();
    opts_->fractal_uv_black = ui->controlBarUVNormMin->GetValue();
    opts_->fractal_uv_white = ui->controlBarUVNormMax->GetValue();
    opts_->fractal_cross_section = ui->checkBoxCorssSection->isChecked();
  }
}

