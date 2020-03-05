#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>

#include <opencv2/opencv.hpp>

#include "backend_api.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget *parent, RenderEngineInterface *engine);
  ~MainWindow();

 protected:
  void showEvent(QShowEvent *ev) override;
  
 private:
  
  void DrawImage(int pos=0);
   
  Ui::MainWindow *ui;
    
  QGraphicsPixmapItem pixmap_;
    
  RenderEngineInterface *engine_;

private slots:
  void onSliderMoved(int position);
};

#endif // MAINWINDOW_H
