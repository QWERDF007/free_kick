#pragma once

#include <QAbstractListModel>
#include <QItemSelectionModel>
#include <QVariantMap>
#include "QuickROI.h"

class QuickROIListModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit QuickROIListModel(QObject *parent = nullptr);
    ~QuickROIListModel();
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
    bool insertRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;
    QHash<int,QByteArray> roleNames() const override;


    Q_INVOKABLE virtual void addROI(const QVariantMap& data);
    Q_INVOKABLE void deleteROI(const QModelIndex &index);



protected:
    std::vector<QuickROI*> rois_;
    QItemSelectionModel *selection_{nullptr};
};
