#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent, RenderEngine *engine) :
    QMainWindow(parent), ui(new Ui::MainWindow), render_watch_(this) {
  ui->setupUi(this);
  engine_ = engine;
  
  // Add a scene and make it show our PixMap
  ui->graphicsView->setScene(new QGraphicsScene(this));
  ui->graphicsView->scene()->addItem(&pixmap_);
  
  engine_options_ = engine_->GetDefaultOptions();
  
  connect(&render_watch_, SIGNAL(finished()), this, SLOT(onRenderFinished()));
  renderFull();
  
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
  qjulia::Image& image = *engine_->Preview(engine_options_);
  QImage qt_image(image.Data()->vals, image.Width(), image.Height(),
                  image.BytesPerRow(), QImage::Format_RGB888);
  
  // QPixmap maintains the image buffer for display.
  // Its data are stored in the graphic card.
  // It seems not be a good solution:
  // TODO: https://stackoverflow.com/questions/33781485/qgraphicspixmapitemsetpixmap-performance
  pixmap_.setPixmap(QPixmap::fromImage(qt_image));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
}

void MainWindow::renderFull(void) {
  render_watch_.cancel();
  render_watch_.waitForFinished();
  QFuture<qjulia::Image*> future = QtConcurrent::run(
    engine_, &RenderEngine::Render, engine_options_);
  render_watch_.setFuture(future);
}
