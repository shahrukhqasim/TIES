package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;

/**
 * Created by srq on 12.10.17.
 */
public class CellBox extends Box {
    static int idNext = 0;
    int id;

    public CellBox(Rectangle2D box) {
        super(box);
        this.id = idNext++;
    }

    public CellBox(Rectangle2D box, int id) {
        super(box);
        this.id = id;
    }

    public int getId() {
        return id;
    }

    @Override
    public Paint getStroke() {
        if (selected)
            return Color.color(0, 0, 1);
        else
            return Color.color(1, 0, 1);
    }
}
