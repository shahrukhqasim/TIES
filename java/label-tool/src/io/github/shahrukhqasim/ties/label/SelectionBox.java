package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;

/**
 * Created by srq on 12.10.17.
 */
public class SelectionBox extends Box {
    public SelectionBox(Rectangle2D box) {
        super(box);
    }

    @Override
    public Paint getStroke() {
        return Color.color(0.9804, 0.4745, 0.2431);
    }
}
