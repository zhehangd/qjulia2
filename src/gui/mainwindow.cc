#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QMessageBox>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent, QJuliaContext *ctx) :
    QMainWindow(parent), ui(new Ui::MainWindow), render_watch_(this) {
  ui->setupUi(this);
  ctx_ = ctx;
  
  // Add a scene and make it show our PixMap
  ui->graphicsView->setScene(new QGraphicsScene(this));
  ui->graphicsView->scene()->addItem(&pixmap_);
  
  scene_params_ = ctx_->GetDefaultOptions();
  
  
  ui->tab_general->AttachContext(ctx_);
  
  ui->tab_general->LinkToOptions(&scene_params_);
  ui->tab_dev->AttachContext(ctx_);
  
  connect(ui->tab_general, SIGNAL(RealtimeParamsChanging(void)), this, SLOT(onRealtimeParamsChanging(void)));
  connect(ui->tab_general, SIGNAL(RealtimeParamsChanged(void)), this, SLOT(onRealtimeParamsChanged(void)));
  connect(ui->tab_general, SIGNAL(Save(void)), this, SLOT(onRenderAndSave(void)));
  connect(ui->tab_dev, SIGNAL(ValueChanging(void)), this, SLOT(onRealtimeParamsChanging(void)));
  connect(ui->tab_dev, SIGNAL(ValueChanged(void)), this, SLOT(onRealtimeParamsChanged(void)));
  connect(ui->actionSave_As, SIGNAL(triggered()), this, SLOT(OnSaveAs()));
  connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(OnSave()));
  connect(&render_watch_, SIGNAL(finished()), this, SLOT(onRenderFinished()));
  onRealtimeParamsChanged();
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::OnSave(void) {
  QString filename;
  if (last_scene_file_.isEmpty()) {
    filename = QFileDialog::getSaveFileName(this, "Save as", {}, "QJulia Scene (*.scene)");
  } else {
    filename = last_scene_file_;
  }
  qDebug() << "Save: " << filename;
  if (!filename.isEmpty()) {
    ctx_->SaveScene(filename);
    last_scene_file_ = filename;
  }
}

void MainWindow::OnSaveAs(void) {
  QString filename = QFileDialog::getSaveFileName(this, "Save as", {}, "QJulia Scene (*.scene)");
  qDebug() << "SaveAs: " << filename;
  if (!filename.isEmpty()) {
    ctx_->SaveScene(filename);
    last_scene_file_ = filename;
  }
}

void MainWindow::onRenderFinished(void) {
  if (render_watch_.isCanceled()) {return;}
  qjulia::Image& image = *render_watch_.result();
  CHECK(image.Data() != nullptr);
  QImage qt_image(image.Data()->vals, image.Width(), image.Height(),
                  image.BytesPerRow(), QImage::Format_RGB888);
  pixmap_.setPixmap(QPixmap::fromImage(qt_image));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}

void MainWindow::DrawImage(void) {
  qjulia::Image& image = *ctx_->FastPreview(scene_params_);
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
  //qDebug() << "Changing";
  render_watch_.cancel();
  render_watch_.waitForFinished();
  DrawImage();
}

void MainWindow::onRealtimeParamsChanged(void) {
  //qDebug() << "Changed";
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<qjulia::Image*> future = QtConcurrent::run(
    ctx_, &QJuliaContext::Render, scene_params_);
  render_watch_.setFuture(future);
}

void MainWindow::onRenderAndSave(void) {
  ctx_->Save(scene_params_);
}
