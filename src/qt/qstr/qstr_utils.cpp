#include "qstr_utils.h"

#include <QStringBuilder>

namespace free_kick::utils::qstr {

QString toQString(const std::vector<double> &vec, const QString &separator)
{
    QString qstr;
    for (double val : vec)
    {
        qstr += QString::number(val) + separator;
    }
    return qstr;
}

QString toQStringv2(const std::vector<double> &vec, const QString &separator)
{
    QString qstr;
    for (double val : vec)
    {
        qstr = qstr % QString::number(val) % separator;
    }
    return qstr;
}
} // namespace free_kick::utils::qstr