#include "widgets/sphericalcoords.h"
#include "sphericalcoordsplugin.h"

#include <QtPlugin>

SphericalCoordsPlugin::SphericalCoordsPlugin(QObject *parent)
    : QObject(parent)
{
    m_initialized = false;
}

void SphericalCoordsPlugin::initialize(QDesignerFormEditorInterface * /* core */)
{
    if (m_initialized)
        return;

    // Add extension registrations, etc. here

    m_initialized = true;
}

bool SphericalCoordsPlugin::isInitialized() const
{
    return m_initialized;
}

QWidget *SphericalCoordsPlugin::createWidget(QWidget *parent)
{
    return new SphericalCoords(parent);
}

QString SphericalCoordsPlugin::name() const
{
    return QLatin1String("SphericalCoords");
}

QString SphericalCoordsPlugin::group() const
{
    return QLatin1String("");
}

QIcon SphericalCoordsPlugin::icon() const
{
    return QIcon();
}

QString SphericalCoordsPlugin::toolTip() const
{
    return QLatin1String("");
}

QString SphericalCoordsPlugin::whatsThis() const
{
    return QLatin1String("");
}

bool SphericalCoordsPlugin::isContainer() const
{
    return false;
}

QString SphericalCoordsPlugin::domXml() const
{
    return QLatin1String("<widget class=\"SphericalCoords\" name=\"sphericalCoords\">\n</widget>\n");
}

QString SphericalCoordsPlugin::includeFile() const
{
    return QLatin1String("widgets/sphericalcoords.h");
}
#if QT_VERSION < 0x050000
Q_EXPORT_PLUGIN2(sphericalcoordsplugin, SphericalCoordsPlugin)
#endif // QT_VERSION < 0x050000
