#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QMessageBox>



MainWindow::MainWindow(QWidget *parent, RenderEngineInterface *engine) :
    QMainWindow(parent), ui(new Ui::MainWindow), render_watch_(this) {
  ui->setupUi(this);
  engine_ = engine;
  
  // Add a scene and make it show our PixMap
  ui->graphicsView->setScene(new QGraphicsScene(this));
  ui->graphicsView->scene()->addItem(&pixmap_);
  
  connect(ui->slider_azi, SIGNAL(valueChanged(int)), this, SLOT(onSliderAziChanged(int)));
  connect(ui->slider_azi, SIGNAL(sliderReleased()), this, SLOT(onSliderAziReleased()));
  connect(ui->slider_alt, SIGNAL(valueChanged(int)), this, SLOT(onSliderAltChanged(int)));
  connect(ui->slider_alt, SIGNAL(sliderReleased()), this, SLOT(onSliderAltReleased()));
  connect(ui->slider_dist, SIGNAL(valueChanged(int)), this, SLOT(onSliderDistChanged(int)));
  connect(ui->slider_dist, SIGNAL(sliderReleased()), this, SLOT(onSliderDistReleased()));
  connect(&render_watch_, SIGNAL(finished()), this, SLOT(onRenderFinished()));
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::showEvent(QShowEvent *ev) {
  DrawImage();
}

void MainWindow::onSliderAziChanged(int position) {
  value_ = position;
  engine_options_.camera_pose[0] = (float)position / 100.0f * 180.0;
  DrawImage(position);
}

void MainWindow::onSliderAziReleased(void) {
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<cv::Mat> future = QtConcurrent::run(engine_, &RenderEngineInterface::Render, engine_options_);
  render_watch_.setFuture(future);
}

void MainWindow::onSliderAltChanged(int position) {
  value_ = position;
  engine_options_.camera_pose[1] = (float)position / 100.0f * 90.0 - 10.0;
  DrawImage(position);
}

void MainWindow::onSliderAltReleased(void) {
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<cv::Mat> future = QtConcurrent::run(engine_, &RenderEngineInterface::Render, engine_options_);
  render_watch_.setFuture(future);
}

void MainWindow::onSliderDistChanged(int position) {
  value_ = position;
  engine_options_.camera_pose[2] = (float)position / 100.0f * 7 + 1;
  DrawImage(position);
}

void MainWindow::onSliderDistReleased(void) {
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<cv::Mat> future = QtConcurrent::run(engine_, &RenderEngineInterface::Render, engine_options_);
  render_watch_.setFuture(future);
}

void MainWindow::onRenderFinished(void) {
  qDebug() << "rendered full";
  cv::Mat image = render_watch_.result();
  QImage qt_image(image.data, image.cols, image.rows,
                  image.step, QImage::Format_RGB888);
  pixmap_.setPixmap(QPixmap::fromImage(qt_image.rgbSwapped()));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}

void MainWindow::DrawImage(int pos) {
  engine_->SetValue((float)pos);
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




