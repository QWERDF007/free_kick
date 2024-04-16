#pragma once

#include <vector>
#include <QList>
#include <QVector>

class QuickROI
{
public:
    explicit QuickROI() {}
    explicit QuickROI(const std::vector<double> &data) : data_{data} {}
    QuickROI(const QuickROI& other) : data_{other.data_} {}
    QuickROI& operator=(const QuickROI& other)
    {
        data_ = other.data_;
        return *this;
    }
    void setData(const QList<double>& data)
    {
        data_ = std::vector<double>(data.constBegin(), data.constEnd());
    }
    void setData(const std::vector<double>& data)
    {
        data_ = data;
    }
    const std::vector<double>& data() const
    {
        return data_;
    }

protected:
    std::vector<double> data_;
};

class QuickRectROI : public QuickROI
{
public:
    explicit QuickRectROI() {}
    explicit QuickRectROI(const std::vector<double> &data) : QuickROI{data} {}
    QuickRectROI(const QuickRectROI& other) : QuickROI{other.data_} {}
    QuickRectROI& operator=(const QuickRectROI& other)
    {
        QuickROI::operator=(other);
        return *this;
    }
    double x() const
    {
        if (data_.empty() || data_.size() < 4)
            return std::nan("-1");
        return data_.at(0);
    }

    double y() const
    {
        if (data_.empty() || data_.size() < 4)
            return std::nan("-1");
        return data_.at(1);
    }

    double width() const
    {
        if (data_.empty() || data_.size() < 4)
            return std::nan("-1");
        return data_.at(2);
    }

    double height() const
    {
        if (data_.empty() || data_.size() < 4)
            return std::nan("-1");
        return data_.at(3);
    }
};
