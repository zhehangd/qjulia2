#include "generalpanel.h"
#include "ui_generalpanel.h"

#include <QDebug>

GeneralPanel::GeneralPanel(QWidget *parent)
    : Panel(parent), ui(new Ui::GeneralPanel) {
  ui->setupUi(this);
  
  connect(ui->lineEditReadtimeResWidth, SIGNAL(editingFinished(void)), this, SLOT(onRealtimeParamsChanged(void)));
  connect(ui->lineEditReadtimeResHeight, SIGNAL(editingFinished(void)), this, SLOT(onRealtimeParamsChanged(void)));
  connect(ui->lineEditReadtimeFastResWidth, SIGNAL(editingFinished(void)), this, SLOT(onRealtimeParamsChanged(void)));
  connect(ui->lineEditReadtimeFastResHeight, SIGNAL(editingFinished(void)), this, SLOT(onRealtimeParamsChanged(void)));
  
  connect(ui->pushButtonOfflineRender, SIGNAL(clicked(bool)), this, SLOT(onSave(bool)));
}

GeneralPanel::~GeneralPanel() {
  delete ui;
}

void GeneralPanel::LinkToOptions(SceneCtrlParams *opts) {
  opts_ = opts;
  ui->lineEditReadtimeResWidth->setText(
    QString::number(opts_->realtime_image_size.width));
  ui->lineEditReadtimeResHeight->setText(
    QString::number(opts_->realtime_image_size.height));
  ui->lineEditReadtimeFastResWidth->setText(
    QString::number(opts_->realtime_fast_image_size.width));
  ui->lineEditReadtimeFastResHeight->setText(
    QString::number(opts_->realtime_fast_image_size.height));
  ui->lineEditOfflineResWidth->setText(
    QString::number(opts_->offline_image_size.width));
  ui->lineEditOfflineResHeight->setText(
    QString::number(opts_->offline_image_size.height));
  ui->lineEditOfflineFilename->setText(
    QString::fromStdString(opts_->offline_filename));
}

void GeneralPanel::CollectParams(void) {
  opts_->realtime_image_size.width = ui->lineEditReadtimeResWidth->text().toInt();
  opts_->realtime_image_size.height = ui->lineEditReadtimeResHeight->text().toInt();
  opts_->realtime_fast_image_size.width = ui->lineEditReadtimeFastResWidth->text().toInt();
  opts_->realtime_fast_image_size.height = ui->lineEditReadtimeFastResHeight->text().toInt();
}

void GeneralPanel::onSave(bool) {
  opts_->offline_image_size.width = ui->lineEditOfflineResWidth->text().toInt();
  opts_->offline_image_size.height = ui->lineEditOfflineResHeight->text().toInt();
  opts_->offline_filename = ui->lineEditOfflineFilename->text().toStdString();
  emit Save();
}
