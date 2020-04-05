#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QFuture>
#include <QtConcurrent>
#include <QFutureWatcher>

#include "engine.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent, RenderEngine *engine);
  ~MainWindow();

private:
  Ui::MainWindow *ui;
    
  QGraphicsPixmapItem pixmap_;
  
  QFutureWatcher<qjulia::Image*> render_watch_;
    
  RenderEngine *engine_;
  
  RenderEngine::SceneOptions engine_options_;
  
  void DrawImage(void);
  
  
private slots:
  // Render a full image
  void renderFull(void);
  
  // Called when engine finishes the rendering,
  // this functions will draw it on the screen.
  void onRenderFinished(void);
};

#endif // MAINWINDOW_H
