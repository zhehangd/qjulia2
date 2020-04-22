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

#include "qjulia_context.h"

int main(int argc, char **argv) {
  QApplication app(argc, argv);
  
  auto engine_sptr = std::make_unique<QJuliaContext>();
  engine_sptr->Init();
  
  MainWindow main_win(nullptr, engine_sptr.get());
  main_win.show();
  return app.exec();
}
