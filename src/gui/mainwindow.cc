#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent, RenderEngineInterface *engine) :
    QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
  engine_ = engine;
  
  // Add a scene and make it show our PixMap
  ui->graphicsView->setScene(new QGraphicsScene(this));
  ui->graphicsView->scene()->addItem(&pixmap_);
  
  connect(ui->horizontalSlider, SIGNAL(sliderMoved(int)), this, SLOT(onSliderMoved(int)));
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::showEvent(QShowEvent *ev) {
  DrawImage();
}

void MainWindow::onSliderMoved(int position) {
  DrawImage(position);
}

void MainWindow::DrawImage(int pos) {
  engine_->SetValue((float)pos);
  cv::Mat &image = engine_->Render();
  QImage qt_image(image.data, image.cols, image.rows,
                  image.step, QImage::Format_RGB888);
  
  // QPixmap maintains the image buffer for display.
  // Its data are stored in the graphic card.
  // It seems not be a good solution:
  // TODO: https://stackoverflow.com/questions/33781485/qgraphicspixmapitemsetpixmap-performance
  pixmap_.setPixmap(QPixmap::fromImage(qt_image.rgbSwapped()));
  ui->graphicsView->fitInView(&pixmap_, Qt::KeepAspectRatio);
  
}