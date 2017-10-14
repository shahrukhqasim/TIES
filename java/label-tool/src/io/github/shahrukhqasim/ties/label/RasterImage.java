package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;

/**
 * Created by srq on 12.10.17.
 */
public class RasterImage implements Drawable {
    Image image;
    Rectangle2D boundingBox;

    public RasterImage(Image image) {
        this.image = image;
        boundingBox = new Rectangle2D(0,0, image.getWidth(), image.getHeight());

    }

    @Override
    public void draw(GraphicsContext graphics, Rectangle2D visibleArea, double scale) {
        double x = visibleArea.getMinX();
        double y = visibleArea.getMinY();
        double width = visibleArea.getMaxX() - x;
        double height = visibleArea.getMaxY() - y;


        for (int i=0;i<30;i++)
            graphics.drawImage(image, x / scale, y / scale, width / scale, height / scale, x, y, width, height);

    }

    @Override
    public Rectangle2D getBoundingBox(double scale) {
        return new Rectangle2D(boundingBox.getMinX() * scale, boundingBox.getMinY() * scale, boundingBox.getMaxX() * scale, boundingBox.getMaxY() * scale);
    }
}
