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
  
  scene_params_ = engine_->GetDefaultOptions();
  
  ui->tab_fractal->LinkToOptions(&scene_params_);
  ui->tab_camera->LinkToOptions(&scene_params_);
  ui->tab_general->LinkToOptions(&scene_params_);
  
  connect(ui->tab_fractal, SIGNAL(RealtimeParamsChanging(void)), this, SLOT(onRealtimeParamsChanging(void)));
  connect(ui->tab_fractal, SIGNAL(RealtimeParamsChanged(void)), this, SLOT(onRealtimeParamsChanged(void)));
  connect(ui->tab_camera, SIGNAL(RealtimeParamsChanging(void)), this, SLOT(onRealtimeParamsChanging(void)));
  connect(ui->tab_camera, SIGNAL(RealtimeParamsChanged(void)), this, SLOT(onRealtimeParamsChanged(void)));
  connect(ui->tab_general, SIGNAL(RealtimeParamsChanging(void)), this, SLOT(onRealtimeParamsChanging(void)));
  connect(ui->tab_general, SIGNAL(RealtimeParamsChanged(void)), this, SLOT(onRealtimeParamsChanged(void)));
  
  connect(ui->tab_general, SIGNAL(Save(void)), this, SLOT(onRenderAndSave(void)));
  
  connect(&render_watch_, SIGNAL(finished()), this, SLOT(onRenderFinished()));
  onRealtimeParamsChanged();
  
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::onRenderFinished(void) {
  qDebug() << "rendered full";
  qjulia::Image& image = *render_watch_.result();
  QImage qt_image(image.Data()->vals, image.Width(), image.Height(),
                  image.BytesPerRow(), QImage::Format_RGB888);
  pixmap_.setPixmap(QPixmap::fromImage(qt_image));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}

void MainWindow::DrawImage(void) {
  qjulia::Image& image = *engine_->Preview(scene_params_);
  QImage qt_image(image.Data()->vals, image.Width(), image.Height(),
                  image.BytesPerRow(), QImage::Format_RGB888);
  
  // QPixmap maintains the image buffer for display.
  // Its data are stored in the graphic card.
  // It seems not be a good solution:
  // TODO: https://stackoverflow.com/questions/33781485/qgraphicspixmapitemsetpixmap-performance
  pixmap_.setPixmap(QPixmap::fromImage(qt_image));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}

void MainWindow::onRealtimeParamsChanging(void) {
  qDebug() << "Changing";
  DrawImage();
}

void MainWindow::onRealtimeParamsChanged(void) {
  qDebug() << "Changed";
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<qjulia::Image*> future = QtConcurrent::run(
    engine_, &RenderEngine::Render, scene_params_);
  render_watch_.setFuture(future);
}

void MainWindow::onRenderAndSave(void) {
  engine_->Save(scene_params_);
}
