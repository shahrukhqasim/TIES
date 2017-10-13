package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Point2D;
import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;

/**
 * Created by srq on 13.10.17.
 */
public class Connection implements Drawable, Selectable {
    Point2D p1;
    Point2D p2;
    boolean selected;
    CellBox nodeA, nodeB;

    public Paint getStroke() {
        if (selected)
            return Color.color(1, 0, 0);
        else
            return Color.color(0, 1, 0);
    }

    public Connection(Point2D p1, Point2D p2, CellBox nodeA, CellBox nodeB) {
        if(p1!=null && p2!=null) {
            this.p1 = p1;
            this.p2 = p2;
            this.nodeA = nodeA;
            this.nodeB = nodeB;
        }
        else
            throw new NullPointerException("Box is null");
    }

    @Override
    public void select(Rectangle2D selectionArea, double scale) {
        Rectangle2D selectionAreaInMemory  = new Rectangle2D(selectionArea.getMinX() / scale, selectionArea.getMinY() / scale, selectionArea.getWidth() / scale, selectionArea.getHeight() / scale);
        if (selectionAreaInMemory.contains(p1) || selectionAreaInMemory.contains(p2))
            selected = true;
        else
            selected = false;
    }

    @Override
    public void draw(GraphicsContext graphics, Rectangle2D visibleArea, double scale) {
        double x = p1.getX();
        double y = p1.getY();
        double x2 = p2.getX();
        double y2 = p2.getY();

        graphics.setStroke(getStroke());
        graphics.strokeLine(x * scale, y * scale, x2 * scale, y2 * scale);
    }

    public void setPoints(Point2D p1, Point2D p2) {
        if(p1!=null && p2!=null) {
            this.p1 = p1;
            this.p2 = p2;
        }
        else
            throw new NullPointerException("Box is null");
    }

    boolean isSelected() {
        return selected;
    }

    @Override
    public Rectangle2D getBoundingBox(double scale) {
        return new Rectangle2D(Math.min(p1.getX(), p2.getX()), Math.min(p1.getY(), p2.getY()), Math.abs(p2.getX() - p1.getX()), Math.abs(p2.getY() - p1.getY()));
    }

    public CellBox getNodeA() {
        return nodeA;
    }

    public CellBox getNodeB() {
        return nodeB;
    }
}
