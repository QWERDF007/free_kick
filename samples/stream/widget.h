#include <vector>

namespace wd {

class Widget
{
private:
    std::vector<int> data_;

public:
    Widget(/* args */);

    Widget(const std::vector<int> &data)
        : data_(data)
    {
    }

    ~Widget();

    void setData(const std::vector<int> &data);

    const std::vector<int> &data() const
    {
        return data_;
    }

    std::vector<int> &data()
    {
        return data_;
    }
};

} // namespace  wd
