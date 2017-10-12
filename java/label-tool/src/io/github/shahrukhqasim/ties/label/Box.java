package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * Created by srq on 12.10.17.
 */
public abstract class Box implements Selectable, Drawable {
    Rectangle2D box;
    boolean selected;

    public abstract Color getStroke();

    public Box(Rectangle2D box) {
        this.box = box;
    }

    @Override
    public void select(Rectangle2D selectionArea, float scale) {
        Rectangle2D selectionAreaInMemory  = new Rectangle2D(selectionArea.getMinX() / scale, selectionArea.getMinY() / scale, selectionArea.getWidth() / scale, selectionArea.getHeight() / scale);
        if (box.intersects(selectionAreaInMemory))
            selected = true;
        else
            selected = false;
    }

    @Override
    public void draw(GraphicsContext graphics, Rectangle2D visibleArea, double scale) {
        double x = box.getMinX();
        double y = box.getMinY();
        double width = box.getMaxX() - x;
        double height = box.getMaxY() - y;

        graphics.setStroke(getStroke());
        graphics.strokeRect(x * scale, y * scale, width * scale, height * scale);
    }

    @Override
    public Rectangle2D getBoundingBox(double scale) {
        return box;
    }
}
