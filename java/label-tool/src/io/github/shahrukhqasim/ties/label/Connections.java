package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.GraphicsContext;

import java.util.Vector;

/**
 * Created by srq on 13.10.17.
 */
public class Connections implements Drawable, Selectable {
    Vector<Connection> connections;

    public Connections(Vector<Connection> connections) {
        this.connections = connections;
    }

    @Override
    public void select(Rectangle2D selectionArea, double scale) {
        for (Connection connection : connections) {
            connection.select(selectionArea, scale);
        }

    }

    public Vector<Connection> getConnections() {
        return connections;
    }

    @Override
    public void draw(GraphicsContext graphics, Rectangle2D visibleArea, double scale) {
        for (Connection connection : connections) {
            connection.draw(graphics, visibleArea, scale);
        }
    }

    @Override
    public Rectangle2D getBoundingBox(double scale) {
        if (connections.size() == 0 ) {
            return new Rectangle2D(0,0,0,0);
        }
        else {
            Rectangle2D boundingBox = connections.get(0).getBoundingBox(scale);
            for (Connection connection : connections) {
                boundingBox = Utils.union(boundingBox, connection.getBoundingBox(scale));
            }
            return boundingBox;
        }
    }
}
