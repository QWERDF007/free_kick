#include "QuickShape.h"
#include <QPainter>

QuickPen::QuickPen(QObject *parent)
    : QObject(parent)
    , width_(0)
    , color_(Qt::black)
    , valid_(false)
    , style_(Qt::PenStyle::SolidLine)
{

}

qreal QuickPen::width() const
{
    return width_;
}

void QuickPen::setWidth(qreal w)
{
    if (w == width_)
        return;
    width_ = w;
    valid_ = color_.alpha() && (qRound(width_) >= 1);
    static_cast<QQuickItem*>(parent())->update();
    emit widthChanged();
}

QColor QuickPen::color() const
{
    return color_;
}

void QuickPen::setColor(const QColor &c)
{
    color_ = c;
    valid_ = color_.alpha() && (qRound(width_) >= 1);
    static_cast<QQuickItem*>(parent())->update();
    emit colorChanged();
}

bool QuickPen::isValid() const
{
    return valid_;
}

Qt::PenStyle QuickPen::style() const
{
    return style_;
}

void QuickPen::setStyle(Qt::PenStyle style)
{
    if (style == style_)
        return;
    style_ = style;
    static_cast<QQuickItem*>(parent())->update();
    emit styleChanged();
}


QuickShape::QuickShape(QQuickItem *parent)
    : QQuickPaintedItem(parent)
    , color_(Qt::white)
{

}

QColor QuickShape::color() const
{
    return color_;
}

void QuickShape::setColor(const QColor &c)
{
    if (color_ == c)
        return;
    color_ = c;
    update();
    emit colorChanged();
}


QuickPen *QuickShape::border()
{
    if (!pen_)
    {
        pen_ = new QuickPen(this);
    }
    return pen_;
}


QuickRectangle::QuickRectangle(QQuickItem *parent)
    : QuickShape(parent)
    , radius_(0)
{

}

qreal QuickRectangle::radius() const
{
    return radius_;
}

void QuickRectangle::setRadius(qreal radius)
{
    if (radius == radius_)
        return;
    radius_ = radius;
    if (radius_ > 0 && !antialiasing())
        setAntialiasing(true);
    update();
    emit radiusChanged();
}


void QuickRectangle::paint(QPainter *painter)
{
    if (antialiasing())
        painter->setRenderHint(QPainter::Antialiasing);
    QRectF r;
    if (pen_ && pen_->isValid())
    {
        QPen pen = painter->pen();
        pen.setColor(pen_->color());
        pen.setWidthF(pen_->width());
        // pen.setCosmetic(true);
        pen.setJoinStyle(Qt::MiterJoin); // 设置连接处 (拐角) 的样式
        pen.setStyle(pen_->style());
        painter->setPen(pen);
        r = QRectF(pen_->width()/2, pen_->width()/2, width() - pen_->width(), height() - pen_->width());
    }
    else
    {
        r = QRectF(0,0, width(), height());
        painter->setPen(Qt::NoPen);
    }
    painter->setBrush(color());
    painter->drawRoundedRect(r, radius_, radius_);
}



QuickCircle::QuickCircle(QQuickItem *parent)
    : QuickShape(parent)
    , radius_(0)
    , center_(QPointF())
{
    setAntialiasing(true);
}

qreal QuickCircle::radius() const
{
    return radius_;
}

void QuickCircle::setRadius(qreal radius)
{
    if (radius == radius_)
        return;
    setWidth(2*radius);
    setHeight(2*radius);
    radius_ = radius;
    update();
    emit radiusChanged();
}

QPointF QuickCircle::center() const
{
    return center_;
}

void QuickCircle::setCenter(const QPointF &center)
{
    if (center == center_)
        return;
    center_ = center;
    setX(center_.x() - radius_);
    setY(center_.y() - radius_);
    qInfo() << __FUNCTION__ << x() << y();
    update();
    emit centerChanged();
}

void QuickCircle::paint(QPainter *painter)
{
    if (antialiasing())
        painter->setRenderHint(QPainter::Antialiasing);
    if (pen_ && pen_->isValid())
    {
        QPen pen = painter->pen();
        pen.setColor(pen_->color());
        pen.setWidthF(pen_->width());
        // pen.setCosmetic(true);
        pen.setJoinStyle(Qt::MiterJoin); // 设置连接处 (拐角) 的样式
        pen.setStyle(pen_->style());
        painter->setPen(pen);
    }
    painter->setBrush(color());
    painter->drawEllipse(QPointF(radius_,radius_), radius_ - border()->width()/2, radius_ - border()->width()/2);
}
