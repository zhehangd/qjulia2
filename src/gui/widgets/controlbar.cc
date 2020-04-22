#include "controlbar.h"
#include "ui_controlbar.h"

#include <cmath>

#include <QDebug>

ControlBar::ControlBar(QWidget *parent) :
    QWidget(parent), ui(new Ui::ControlBar) {
  ui->setupUi(this);
  value_changed_timer_ = new QTimer(this);
  connect(ui->slider, SIGNAL(actionTriggered(int)), this, SLOT(onSliderActed(int)));
  connect(ui->edit, SIGNAL(editingFinished(void)), this, SLOT(onEditChanged(void)));
  connect(value_changed_timer_, SIGNAL(timeout(void)), this, SLOT(SendValueChangedSignal(void)));
  
  ShowSliderPosition();
  ShowEditContent();
}

void ControlBar::SetValue(float value) {
  value_ = value;
  ShowSliderPosition();
  ShowEditContent();
}

void ControlBar::SetValueRange(float from, float to) {
  vsrt_ = from;
  vend_ = to;
  ShowSliderPosition();
}

void ControlBar::SetLogScale(bool enable) {
  use_log_ = enable;
  ShowSliderPosition();
}


bool ControlBar::GetLogScale(void) const {
  return use_log_;
}

QString ControlBar::GetName(void) const {
  return ui->name->text();
}

void ControlBar::SetName(QString name) {
  ui->name->setText(name);
}

int ControlBar::GetNameWidth(void) const {
  return ui->name->minimumWidth();
}

void ControlBar::SetNameWidth(int w) {
  ui->name->setMinimumWidth(w);
}

void ControlBar::onSliderActed(int action) {
  (void)action;
  value_ = TickToValue(ui->slider->sliderPosition());
  ShowEditContent();
  emit valueChanging(value_);
  TimeForValueChangedSignal();
}

void ControlBar::onEditChanged(void) {
  value_ = ui->edit->text().toFloat();
  ShowEditContent();
  ShowSliderPosition();
  emit valueChanging(value_);
  TimeForValueChangedSignal();
}

ControlBar::~ControlBar() {
  delete ui;
}

void ControlBar::ShowEditContent(void) {
   ui->edit->setText(QString("%1").arg(value_, 0, 'g', 2));
}

void ControlBar::ShowSliderPosition(void) {
  ui->slider->setValue(ValueToTick(value_));
}

void ControlBar::TimeForValueChangedSignal(void) {
  value_changed_timer_->start(500);
}

float ControlBar::TickToValue(int tick) const {
  if (use_log_) {
    float p = (float)(tick - tsrt_) / (tend_ - tsrt_);
    float v = vsrt_ * std::pow(vend_ / vsrt_, p);
    return v;  
  } else {
    float v = (float)(tick - tsrt_) / (tend_ - tsrt_) * (vend_ - vsrt_) + vsrt_;
    return v;
  }
}

int ControlBar::ValueToTick(float value) const {
  if (value > vend_) {value = vend_;}
  if (value < vsrt_) {value = vsrt_;}
  if (use_log_) {
    float k = std::log(value / vsrt_) / std::log(vend_ / vsrt_);
    int t = k * (tend_ - tsrt_) + tsrt_;
    return t;
  } else {
    int t = (int)((value - vsrt_) / (vend_ - vsrt_) * (tend_ - tsrt_) + tsrt_);
    return t;
  }
}

void ControlBar::SendValueChangedSignal(void) {
  value_changed_timer_->stop();
  emit valueChanged(value_);
}
