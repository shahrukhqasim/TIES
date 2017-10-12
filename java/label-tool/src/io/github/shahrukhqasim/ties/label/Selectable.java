package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;

/**
 * Created by srq on 12.10.17.
 */
public interface Selectable extends Drawable {
    /**
     * If the user tries to select it, the selection area is passed to this function
     *
     * @param selectionArea The area user selects
     * @param scale Scale (zoom level) selected by the user
     */
    void select(Rectangle2D selectionArea, float scale);
}
