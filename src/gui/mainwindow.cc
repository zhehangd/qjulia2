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
  
  connect(ui->horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(onSliderValueChanged(int)));
  connect(ui->horizontalSlider, SIGNAL(sliderReleased()), this, SLOT(onSliderReleased()));
  connect(&render_watch_, SIGNAL(finished()), this, SLOT(onRenderFinished()));
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::showEvent(QShowEvent *ev) {
  DrawImage();
}

void MainWindow::onSliderValueChanged(int position) {
  qDebug() << "value changed";
  value_ = position;
  DrawImage(position);
}

void MainWindow::onSliderReleased(void) {
  qDebug() << "released";
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<cv::Mat> future = QtConcurrent::run(engine_, &RenderEngineInterface::Render);
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
  cv::Mat image = engine_->Preview();
  QImage qt_image(image.data, image.cols, image.rows,
                  image.step, QImage::Format_RGB888);
  
  // QPixmap maintains the image buffer for display.
  // Its data are stored in the graphic card.
  // It seems not be a good solution:
  // TODO: https://stackoverflow.com/questions/33781485/qgraphicspixmapitemsetpixmap-performance
  pixmap_.setPixmap(QPixmap::fromImage(qt_image.rgbSwapped()));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}




