#include "widget.h"

namespace wd {

Widget::Widget(/* args */) {}

Widget::~Widget() {}

void Widget::setData(const std::vector<int> &data)
{
    data_ = data;
}

} // namespace wd
