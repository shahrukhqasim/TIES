package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * Created by srq on 12.10.17.
 */
public class OcrBox extends Box {

    public OcrBox(Rectangle2D box) {
        super(box);
    }

    @Override
    public Color getStroke() {
        if (selected)
            return Color.color(1, 0, 0);
        else
            return Color.color(0, 1, 0);
    }
}
