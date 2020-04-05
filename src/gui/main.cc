#include <memory>

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

#include "mainwindow.h"

#include "engine.h"

int main(int argc, char **argv) {
  QApplication app(argc, argv);
  
  auto engine_sptr = std::make_unique<RenderEngine>();
  engine_sptr->Init("../data/julia.scene");
  
  MainWindow main_win(nullptr, engine_sptr.get());
  main_win.show();
  return app.exec();
}
