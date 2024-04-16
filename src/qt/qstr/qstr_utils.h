#pragma once
#include <QString>

namespace free_kick::utils::qstr {

QString toQString(const std::vector<double> &vec, const QString &separator = ", ");

} // namespace free_kick::utils::qstr
