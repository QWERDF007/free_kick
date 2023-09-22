#include "widget.h"

#include <iostream>
#include <type_traits>
#include <vector>

template<typename T>
struct is_Widget
{
    static const bool value = std::is_base_of<wd::Widget, T>::value;
};

template<typename T>
struct is_builtin
{
    static const bool value = std::is_arithmetic<T>::value || std::is_void<T>::value;
};

template<typename T>
typename std::enable_if<is_Widget<T>::value, std::ostream &>::type operator<<(std::ostream         &os,
                                                                              const std::vector<T> &vec)
// std::ostream &operator<<(std::ostream &os, const std::vector<wd::Widget> &vec)
{
    os << "[";
    for (int i = 0; i < vec.size(); ++i)
    {
        os << vec[i].data();
        if (i != vec.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template<typename T>
typename std::enable_if<is_builtin<T>::value, std::ostream &>::type operator<<(std::ostream         &os,
                                                                               const std::vector<T> &vec)
{
    os << "[";
    for (int i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i != vec.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

int main(int argc, char **argv)
{
    std::cout << "hello world\n";
    std::vector<int>        a{1, 2, 3};
    std::vector<int>        a2{4, 5, 6};
    // std::vector<double>     b{1.5, 2.9, 4.0};
    std::vector<wd::Widget> c;
    c.emplace_back(wd::Widget(a));
    c.emplace_back(wd::Widget(a2));
    // std::cout << a << std::endl;
    // std::cout << b << std::endl;
    std::cout << c << std::endl;
    // std::cout << is_Widget<wd::Widget>::value << std::endl;
    return 0;
}