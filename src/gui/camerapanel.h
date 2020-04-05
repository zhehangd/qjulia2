#ifndef CAMERAPANEL_H
#define CAMERAPANEL_H

#include <QWidget>

namespace Ui {
class CameraPanel;
}

class CameraPanel : public QWidget
{
    Q_OBJECT

public:
    explicit CameraPanel(QWidget *parent = 0);
    ~CameraPanel();

private:
    Ui::CameraPanel *ui;
};

#endif // CAMERAPANEL_H
