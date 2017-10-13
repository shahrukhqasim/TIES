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
    public void select(Rectangle2D selectionArea, double scale) {
        for (Box box : boxes) {
            box.select(selectionArea, scale);
        }

    }

    public Vector<Box> getBoxes() {
        return boxes;
    }

    @Override
    public void draw(GraphicsContext graphics, Rectangle2D visibleArea, double scale) {
        for (Box box : boxes) {
            box.draw(graphics, visibleArea, scale);
        }
    }

    @Override
    public Rectangle2D getBoundingBox(double scale) {
        if (boxes.size() == 0 ) {
            return new Rectangle2D(0,0,0,0);
        }
        else {
            Rectangle2D boundingBox = boxes.get(0).getBoundingBox(scale);
            for (Box box : boxes) {
                boundingBox = Utils.union(boundingBox, box.getBoundingBox(scale));
            }
            return boundingBox;
        }
    }
}
