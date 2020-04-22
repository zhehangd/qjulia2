#include "widgets/xyzcoords.h"
#include "xyzcoordsplugin.h"

#include <QtPlugin>

XyzCoordsPlugin::XyzCoordsPlugin(QObject *parent)
    : QObject(parent)
{
    m_initialized = false;
}

void XyzCoordsPlugin::initialize(QDesignerFormEditorInterface * /* core */)
{
    if (m_initialized)
        return;

    // Add extension registrations, etc. here

    m_initialized = true;
}

bool XyzCoordsPlugin::isInitialized() const
{
    return m_initialized;
}

QWidget *XyzCoordsPlugin::createWidget(QWidget *parent)
{
    return new XyzCoords(parent);
}

QString XyzCoordsPlugin::name() const
{
    return QLatin1String("XyzCoords");
}

QString XyzCoordsPlugin::group() const
{
    return QLatin1String("");
}

QIcon XyzCoordsPlugin::icon() const
{
    return QIcon();
}

QString XyzCoordsPlugin::toolTip() const
{
    return QLatin1String("");
}

QString XyzCoordsPlugin::whatsThis() const
{
    return QLatin1String("");
}

bool XyzCoordsPlugin::isContainer() const
{
    return false;
}

QString XyzCoordsPlugin::domXml() const
{
    return QLatin1String("<widget class=\"XyzCoords\" name=\"xyzCoords\">\n</widget>\n");
}

QString XyzCoordsPlugin::includeFile() const
{
    return QLatin1String("widgets/xyzcoords.h");
}
#if QT_VERSION < 0x050000
Q_EXPORT_PLUGIN2(xyzcoordsplugin, XyzCoordsPlugin)
#endif // QT_VERSION < 0x050000
