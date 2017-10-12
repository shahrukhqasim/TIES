package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.GraphicsContext;

/**
 * Created by srq on 12.10.17.
 */
public interface Drawable {
    /**
     * Draw the drawable object in accord with the scale.
     *  @param graphics The graphics object used to draw
     * @param visibleArea Scaled area
     * @param scale Scale selected
     */
    void draw(GraphicsContext graphics, Rectangle2D visibleArea, double scale);

    /**
     * To get the scaled bounding box used for drawing
     *
     * @return the bounding box
     * @param scale
     */
    Rectangle2D getBoundingBox(double scale);
}
