#pragma once

#include <QObject>
#include <QQmlEngine>
#include <QQuickPaintedItem>
#include <QColor>
#include <QPen>

class QuickPen : public QObject
{
    Q_OBJECT
    Q_PROPERTY(qreal width READ width WRITE setWidth NOTIFY widthChanged FINAL)
    Q_PROPERTY(QColor color READ color WRITE setColor NOTIFY colorChanged FINAL)
    Q_PROPERTY(Qt::PenStyle style READ style WRITE setStyle NOTIFY styleChanged FINAL)
    QML_ANONYMOUS // QML 可用, 但无法从 QML 定义, 只能从 C++ 层传递
public:
    explicit QuickPen(QObject *parent=nullptr);

    qreal width() const;
    void setWidth(qreal w);

    QColor color() const;
    void setColor(const QColor &c);

    bool isValid() const;

    Qt::PenStyle style() const;
    void setStyle(Qt::PenStyle);

private:
    qreal width_{0};
    QColor color_{Qt::white};
    bool valid_{false};
    Qt::PenStyle style_{Qt::PenStyle::SolidLine};

signals:
    void widthChanged();
    void colorChanged();
    void styleChanged();
};

class QuickShape : public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY(QColor color READ color WRITE setColor NOTIFY colorChanged)
    Q_PROPERTY(QuickPen * border READ border CONSTANT FINAL)
public:
    explicit QuickShape(QQuickItem *parent = nullptr);

    QColor color() const;
    void setColor(const QColor &);

    QuickPen *border();

protected:
    QColor color_{Qt::white};
    QuickPen *pen_{nullptr};

signals:
    void colorChanged();

};


class QuickRectangle : public QuickShape
{
    Q_OBJECT
    Q_PROPERTY(qreal radius READ radius WRITE setRadius NOTIFY radiusChanged FINAL)
    QML_NAMED_ELEMENT(QuickRectangle)
public:

    explicit QuickRectangle(QQuickItem *parent = nullptr);

    qreal radius() const;
    void setRadius(qreal radius);

protected:
    void paint(QPainter *painter) override;

private:
    qreal radius_{0};

signals:
    void radiusChanged();
};

class QuickCircle : public QuickShape
{
    Q_OBJECT
    Q_PROPERTY(qreal radius READ radius WRITE setRadius NOTIFY radiusChanged FINAL)
    Q_PROPERTY(QPointF center READ center WRITE setCenter NOTIFY radiusChanged FINAL)
    QML_NAMED_ELEMENT(QuickCircle)
public:
    explicit QuickCircle(QQuickItem *parent = nullptr);

    qreal radius() const;
    void setRadius(qreal radius);

    QPointF center() const;
    void setCenter(const QPointF& center);

protected:
    void paint(QPainter *painter) override;

private:
    qreal radius_{0};
    QPointF center_;

signals:
    void radiusChanged();
    void centerChanged();
};
