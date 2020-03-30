#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QMessageBox>



MainWindow::MainWindow(QWidget *parent, RenderEngine *engine) :
    QMainWindow(parent), ui(new Ui::MainWindow), render_watch_(this) {
  ui->setupUi(this);
  engine_ = engine;
  
  // Add a scene and make it show our PixMap
  ui->graphicsView->setScene(new QGraphicsScene(this));
  ui->graphicsView->scene()->addItem(&pixmap_);
  
  
  
  slider_azi_cvt_.vsrt = 0;
  slider_azi_cvt_.vend = 180;
  slider_alt_cvt_.vsrt = -10;
  slider_alt_cvt_.vend = 90;
  slider_dist_cvt_.vsrt = 8;
  slider_dist_cvt_.vend = 1;
  slider_jconst_cvt_.vsrt = -1;
  slider_jconst_cvt_.vend = 1;
  slider_precision_cvt_.vsrt = 1e-3;
  slider_precision_cvt_.vend = 1e-5;
  
  engine_options_ = engine_->GetDefaultOptions();
  ui->slider_azi->setValue(slider_azi_cvt_.ValueToTick(engine_options_.camera_pose[0]));
  ui->slider_alt->setValue(slider_alt_cvt_.ValueToTick(engine_options_.camera_pose[1]));
  ui->slider_dist->setValue(slider_dist_cvt_.ValueToTick(engine_options_.camera_pose[2]));
  ui->slider_const1->setValue(slider_jconst_cvt_.ValueToTick(engine_options_.julia_constant[0]));
  ui->slider_const2->setValue(slider_jconst_cvt_.ValueToTick(engine_options_.julia_constant[1]));
  ui->slider_const3->setValue(slider_jconst_cvt_.ValueToTick(engine_options_.julia_constant[2]));
  ui->slider_const4->setValue(slider_jconst_cvt_.ValueToTick(engine_options_.julia_constant[3]));
  ui->slider_precision->setValue(slider_precision_cvt_.ValueToTick(engine_options_.precision));
  
  connect(ui->slider_azi, SIGNAL(valueChanged(int)), this, SLOT(onSliderAziChanged(int)));
  connect(ui->slider_azi, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(ui->slider_alt, SIGNAL(valueChanged(int)), this, SLOT(onSliderAltChanged(int)));
  connect(ui->slider_alt, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(ui->slider_dist, SIGNAL(valueChanged(int)), this, SLOT(onSliderDistChanged(int)));
  connect(ui->slider_dist, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(ui->slider_const1, SIGNAL(valueChanged(int)), this, SLOT(onSliderJConst1Changed(int)));
  connect(ui->slider_const1, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(ui->slider_const2, SIGNAL(valueChanged(int)), this, SLOT(onSliderJConst2Changed(int)));
  connect(ui->slider_const2, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(ui->slider_const3, SIGNAL(valueChanged(int)), this, SLOT(onSliderJConst3Changed(int)));
  connect(ui->slider_const3, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(ui->slider_const4, SIGNAL(valueChanged(int)), this, SLOT(onSliderJConst4Changed(int)));
  connect(ui->slider_const4, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(ui->slider_precision, SIGNAL(valueChanged(int)), this, SLOT(onSliderPrecisionChanged(int)));
  connect(ui->slider_precision, SIGNAL(sliderReleased()), this, SLOT(renderFull()));
  connect(&render_watch_, SIGNAL(finished()), this, SLOT(onRenderFinished()));
  renderFull();
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::showEvent(QShowEvent *ev) {
  (void)ev;
}

void MainWindow::renderFull(void) {
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<cv::Mat> future = QtConcurrent::run(engine_, &RenderEngine::Render, engine_options_);
  render_watch_.setFuture(future);
}

void MainWindow::onSliderAltChanged(int position) {
  engine_options_.camera_pose[1] = slider_alt_cvt_.TickToValue(position);
  DrawImage();
}

void MainWindow::onSliderAziChanged(int position) {
  engine_options_.camera_pose[0] = slider_azi_cvt_.TickToValue(position);
  DrawImage();
}

void MainWindow::onSliderDistChanged(int position) {
  engine_options_.camera_pose[2] = slider_dist_cvt_.TickToValue(position);
  DrawImage();
}

void MainWindow::onSliderJConst1Changed(int position) {
  float val = slider_jconst_cvt_.TickToValue(position);
  engine_options_.julia_constant[0] = val;
  ui->label_const1->setText(QString("%1").arg(val, 0, 'g', 2));
  DrawImage();
}

void MainWindow::onSliderJConst2Changed(int position) {
  float val = slider_jconst_cvt_.TickToValue(position);
  engine_options_.julia_constant[1] = val;
  ui->label_const2->setText(QString("%1").arg(val, 0, 'g', 2));
  DrawImage();
}

void MainWindow::onSliderJConst3Changed(int position) {
  float val = slider_jconst_cvt_.TickToValue(position);
  engine_options_.julia_constant[2] = val;
  ui->label_const3->setText(QString("%1").arg(val, 0, 'g', 2));
  DrawImage();
}

void MainWindow::onSliderJConst4Changed(int position) {
  float val = slider_jconst_cvt_.TickToValue(position);
  engine_options_.julia_constant[3] = val;
  ui->label_const4->setText(QString("%1").arg(val, 0, 'g', 2));
  DrawImage();
}

void MainWindow::onSliderPrecisionChanged(int position) {
  float val = slider_precision_cvt_.TickToValue(position);
  engine_options_.precision = val;
  ui->label_precision->setText(QString("%1").arg(val, 0, 'e', 2));
  DrawImage();
}

void MainWindow::onRenderFinished(void) {
  qDebug() << "rendered full";
  cv::Mat image = render_watch_.result();
  QImage qt_image(image.data, image.cols, image.rows,
                  image.step, QImage::Format_RGB888);
  pixmap_.setPixmap(QPixmap::fromImage(qt_image.rgbSwapped()));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}

void MainWindow::DrawImage(void) {
  cv::Mat image = engine_->Preview(engine_options_);
  QImage qt_image(image.data, image.cols, image.rows,
                  image.step, QImage::Format_RGB888);
  
  // QPixmap maintains the image buffer for display.
  // Its data are stored in the graphic card.
  // It seems not be a good solution:
  // TODO: https://stackoverflow.com/questions/33781485/qgraphicspixmapitemsetpixmap-performance
  pixmap_.setPixmap(QPixmap::fromImage(qt_image.rgbSwapped()));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}




