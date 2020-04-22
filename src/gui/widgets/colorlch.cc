#include "colorlch.h"
#include "ui_colorlch.h"

ColorLCH::ColorLCH(QWidget *parent) :
    QWidget(parent), ui(new Ui::ColorLCH) {
  ui->setupUi(this);
  auto send_changing = [&]() {emit ValueChanging();};
  auto send_changed = [&]() {emit ValueChanged();};
  connect(ui->l, &ControlBar::valueChanging, send_changing);
  connect(ui->c, &ControlBar::valueChanging, send_changing);
  connect(ui->h, &ControlBar::valueChanging, send_changing);
  connect(ui->l, &ControlBar::valueChanged, send_changed);
  connect(ui->c, &ControlBar::valueChanged, send_changed);
  connect(ui->h, &ControlBar::valueChanged, send_changed);
}

ColorLCH::~ColorLCH() {
  delete ui;
}

#ifndef QJULIA_QT_PLUGIN
qjulia::Vector3f ColorLCH::GetLCHColor(void) const {
  return {ui->l->GetValue(), ui->c->GetValue(), ui->h->GetValue()};
}

void ColorLCH::SetLCHColor(qjulia::Vector3f v) {
  ui->l->SetValue(v[0]);
  ui->c->SetValue(v[1]);
  ui->h->SetValue(v[2]);
}

qjulia::Vector3f ColorLCH::GetRGBColor(void) const {
  return qjulia::LCH2RGB(GetLCHColor());
}

void ColorLCH::SetRGBColor(qjulia::Vector3f v) {
  SetLCHColor(qjulia::RGB2LCH(v));
}
#endif
