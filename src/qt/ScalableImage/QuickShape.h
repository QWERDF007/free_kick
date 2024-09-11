#pragma once

#include <QColor>
#include <QObject>
#include <QPen>
#include <QQmlEngine>
#include <QQuickPaintedItem>

class QuickPen : public QObject
{
    Q_OBJECT
    // QML 可用, 但无法从 QML 定义, 只能从 C++ 层传递
    QML_ANONYMOUS
    Q_PROPERTY(qreal width READ width WRITE setWidth NOTIFY widthChanged FINAL)
    Q_PROPERTY(QColor color READ color WRITE setColor NOTIFY colorChanged FINAL)
    Q_PROPERTY(Qt::PenStyle style READ style WRITE setStyle NOTIFY styleChanged FINAL)
    Q_PROPERTY(Qt::PenJoinStyle joinStyle READ joinStyle WRITE setJoinStyle NOTIFY joinStyleChanged FINAL)

public:
    explicit QuickPen(QObject *parent = nullptr);

    qreal width() const;
    void  setWidth(qreal w);

    QColor color() const;
    void   setColor(const QColor &c);

    bool isValid() const;

    Qt::PenStyle style() const;
    void         setStyle(Qt::PenStyle);

    Qt::PenJoinStyle joinStyle() const;
    void             setJoinStyle(Qt::PenJoinStyle);

private:
    qreal            width_{0};
    QColor           color_{Qt::white};
    bool             valid_{false};
    Qt::PenStyle     style_{Qt::PenStyle::SolidLine};
    Qt::PenJoinStyle join_style_{Qt::MiterJoin};

signals:
    void widthChanged();
    void colorChanged();
    void styleChanged();
    void joinStyleChanged();
};

class QuickShape : public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY(QColor color READ color WRITE setColor NOTIFY colorChanged)
    Q_PROPERTY(QuickPen *border READ border CONSTANT FINAL)
    Q_PROPERTY(QuickShapeType shapeType READ shapeType CONSTANT FINAL)
    QML_NAMED_ELEMENT(QuickShape)
    QML_UNCREATABLE("Can't not create a QuickShape directly")
public:
    explicit QuickShape(QQuickItem *parent = nullptr);

    QColor color() const;
    void   setColor(const QColor &);

    QuickPen *border();

    enum QuickShapeType
    {
        NoShape   = 0x00000000,
        Rectangle = 0x00000001,
        Circle    = 0x00000002,
        Polygon   = 0x00000004,
        AllShapes = 0x07ffffff,
    };
    Q_ENUM(QuickShapeType)

    virtual QuickShapeType shapeType() const = 0;

protected:
    QColor    color_{Qt::white};
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
    void  setRadius(qreal radius);

    QuickShapeType shapeType() const override
    {
        return QuickShapeType::Rectangle;
    }

    void paint(QPainter *painter) override;

private:
    qreal radius_{0};

signals:
    void radiusChanged();
};

class QuickCenter : public QObject
{
    Q_OBJECT
    Q_PROPERTY(qreal x READ x WRITE setX NOTIFY xChanged FINAL)
    Q_PROPERTY(qreal y READ y WRITE setY NOTIFY yChanged FINAL)
    Q_PROPERTY(qreal size READ size WRITE setSize NOTIFY sizeChanged FINAL)
    QML_ANONYMOUS
public:
    explicit QuickCenter(QObject *parent = nullptr);

    qreal x() const;
    void  setX(qreal);

    qreal y() const;
    void  setY(qreal);

    qreal size() const;
    void  setSize(qreal);

private:
    qreal x_{0};
    qreal y_{0};
    qreal size_{0};

signals:
    void xChanged();
    void yChanged();
    void sizeChanged();
};

class QuickCircle : public QuickShape
{
    Q_OBJECT
    Q_PROPERTY(qreal radius READ radius WRITE setRadius NOTIFY radiusChanged FINAL)
    Q_PROPERTY(QPointF center READ center WRITE setCenter NOTIFY radiusChanged FINAL)
    Q_PROPERTY(bool centerVisible READ centerVisible WRITE setCenterVisible NOTIFY centerVisibleChanged)
    QML_NAMED_ELEMENT(QuickCircle)
public:
    explicit QuickCircle(QQuickItem *parent = nullptr);

    qreal radius() const;
    void  setRadius(qreal radius);

    QPointF center() const;
    void    setCenter(const QPointF &center);

    bool centerVisible() const;
    void setCenterVisible(const bool);

    QuickShapeType shapeType() const override
    {
        return QuickShapeType::Circle;
    }

    void paint(QPainter *painter) override;

private:
    qreal   radius_{0};
    QPointF center_;
    bool    center_visible_{true};

    //    QuickCenter *center_{nullptr};

private slots:
    void updateOnXChanged();
    void updateOnYChanged();

signals:
    void radiusChanged();
    void centerChanged();
    void centerVisibleChanged();
};

class QuickPolygon : public QuickShape
{
    Q_OBJECT
    Q_PROPERTY(QList<QPointF> points READ points WRITE setPoints NOTIFY pointsChanged)
    QML_NAMED_ELEMENT(QuickPolygon)
public:
    explicit QuickPolygon(QQuickItem *parent = nullptr);

    QList<QPointF> points() const;
    void           setPoints(const QList<QPointF> &points);

    QuickShapeType shapeType() const override
    {
        return QuickShapeType::Polygon;
    }

    void paint(QPainter *painter) override;

private:
    QList<QPointF> points_;

signals:
    void pointsChanged();
};
