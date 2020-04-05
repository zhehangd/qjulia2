#include "generalpanel.h"
#include "ui_generalpanel.h"

#include <QDebug>

GeneralPanel::GeneralPanel(QWidget *parent)
    : Panel(parent), ui(new Ui::GeneralPanel) {
  ui->setupUi(this);
  
  connect(ui->pushButtonOfflineRender, SIGNAL(clicked(bool)), this, SLOT(onSave(bool)));
}

GeneralPanel::~GeneralPanel() {
  delete ui;
}

void GeneralPanel::LinkToOptions(SceneCtrlParams *opts) {
  opts_ = opts;
  ui->lineEditOfflineResWidth->setText(
    QString::number(opts_->offline_image_size.width));
  ui->lineEditOfflineResHeight->setText(
    QString::number(opts_->offline_image_size.height));
  ui->lineEditOfflineFilename->setText(
    QString::fromStdString(opts_->offline_filename));
}

void GeneralPanel::CollectParams(void) {
}

void GeneralPanel::onSave(bool) {
  opts_->offline_image_size.width = ui->lineEditOfflineResWidth->text().toInt();
  opts_->offline_image_size.height = ui->lineEditOfflineResHeight->text().toInt();
  opts_->offline_filename = ui->lineEditOfflineFilename->text().toStdString();
  emit Save();
}
