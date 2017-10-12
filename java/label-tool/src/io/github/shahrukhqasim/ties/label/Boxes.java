package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.GraphicsContext;

import java.util.Vector;

/**
 * Created by srq on 12.10.17.
 */
public class Boxes implements Drawable, Selectable {
    Vector<Box> boxes;

    public Boxes(Vector<Box> boxes) {
        this.boxes = boxes;
    }

    @Override
    public void select(Rectangle2D selectionArea, float scale) {
        for (Box box : boxes) {
            box.select(selectionArea, scale);
        }

    }

    @Override
    public void draw(GraphicsContext graphics, Rectangle2D visibleArea, double scale) {
        for (Box box : boxes) {
            box.draw(graphics, visibleArea, scale);
        }
    }

    public Rectangle2D union(Rectangle2D a, Rectangle2D b) {
        double x1 = Math.min(a.getMinX(), b.getMinX());
        double y1 = Math.min(a.getMinY(), b.getMinY());
        double x2 = Math.min(a.getMaxX(), b.getMaxX());
        double y2 = Math.min(a.getMaxY(), b.getMaxY());
        return new Rectangle2D(x1, y1, x2 - x1, y2 - y1);
    }

    @Override
    public Rectangle2D getBoundingBox(double scale) {
        if (boxes.size() == 0 ) {
            return new Rectangle2D(0,0,0,0);
        }
        else {
            Rectangle2D boundingBox = boxes.get(0).getBoundingBox(scale);
            for (Box box : boxes) {
                boundingBox = union(boundingBox, box.getBoundingBox(scale));
            }
            return boundingBox;
        }
    }
}
