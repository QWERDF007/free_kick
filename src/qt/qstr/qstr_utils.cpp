#include "qstr_utils.h"

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
} // namespace free_kick::utils::qstr