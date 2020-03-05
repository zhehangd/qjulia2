#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QSizePolicy>
#include <QSlider>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QImage>
#include <QMessageBox>

#include <opencv2/opencv.hpp>

#include "mainwindow.h"

#include "engine.h"

int main(int argc, char **argv) {
  QApplication app(argc, argv);
  
  if (argc <= 1) {
    std::cerr << "usage: qjulia2 [SCENE-FILE]" << std::endl;
    return 1;
  }
  
  auto api = CreateDefaultEngine();
  api->Init(argv[1]);
  MainWindow main_win(nullptr, api.get());
  main_win.show();
  return app.exec();
}
