#include "QuickROIListModel.h"

enum QuickROIRole
{
    QuickROIX = Qt::UserRole + 1,
    QuickROIY = Qt::UserRole + 1,
    QuickROIWidth = Qt::UserRole + 1,
    QuickROIHeight = Qt::UserRole + 1,
    QuickROISelected = Qt::UserRole + 1,
};

QuickROIListModel::QuickROIListModel(QObject *parent)
    : QAbstractListModel{parent}
    , selection_{new QItemSelectionModel(this, this)}
{
}

QuickROIListModel::~QuickROIListModel()
{
    for (QuickROI* roi : rois_)
    {
        if (roi)
        {
            delete roi;
            roi = nullptr;
        }
    }
}

int QuickROIListModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;
    return rois_.size();
}

QVariant QuickROIListModel::data(const QModelIndex &index, int role) const
{
    return QVariant();
}

bool QuickROIListModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    return false;
}

bool QuickROIListModel::insertRows(int row, int count, const QModelIndex &parent)
{
    if (count < 1 || row < 0 || row > rowCount(parent))
        return false;

    beginInsertRows(QModelIndex(), row, row + count - 1);

    // do something
    rois_.insert(rois_.begin() + row, count, new QuickRectROI);

    endInsertRows();

    return true;
}

bool QuickROIListModel::removeRows(int row, int count, const QModelIndex &parent)
{
    if (count <= 0 || row < 0 || (row + count) > rowCount(parent))
        return false;

    beginRemoveRows(QModelIndex(), row, row + count - 1);

    const auto it = rois_.begin() + row;
    rois_.erase(it, it + count);

    endRemoveRows();

    return true;
}

QHash<int, QByteArray> QuickROIListModel::roleNames() const
{
    auto roles = QAbstractListModel::roleNames();
    roles.insert(QuickROIX, "x");
    roles.insert(QuickROIY, "y");
    roles.insert(QuickROIWidth, "width");
    roles.insert(QuickROIHeight, "height");
    return roles;
}

void QuickROIListModel::addROI(const QVariantMap &data)
{
    int row = rowCount();
    insertRow(row);
    auto _index = index(row);
    setData(_index, data["x"], QuickROIX);
    setData(_index, data["y"], QuickROIY);
    setData(_index, data["width"], QuickROIWidth);
    setData(_index, data["height"], QuickROIHeight);
}

void QuickROIListModel::deleteROI(const QModelIndex &index)
{
    removeRow(index.row());
}
