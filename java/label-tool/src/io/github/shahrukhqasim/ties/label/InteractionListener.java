package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Point2D;
import javafx.geometry.Rectangle2D;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;

/**
 * Created by srq on 12.10.17.
 */
public interface InteractionListener {
    void drag(Point2D startPoint, Point2D endPoint, double scale);
    void dragReleased(Rectangle2D rectangle, double scale, MouseButton button);
    void keyPressed(double scale, KeyCode keyCode);
}
