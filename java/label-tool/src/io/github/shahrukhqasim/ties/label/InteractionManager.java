package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Point2D;
import javafx.geometry.Rectangle2D;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;

import java.security.Key;
import java.util.Vector;

/**
 * Created by srq on 12.10.17.
 */
public class InteractionManager implements InteractionListener {
    SelectionBox selectionBox;
    Boxes boxesOcr;
    Boxes boxesCell;
    boolean connectMode = false;
    Connection selectionConnection;
    Connections connections;


    public InteractionManager(Boxes boxesOcr, Boxes boxesCells, Connections connections) {
        selectionBox = new SelectionBox(new Rectangle2D(0,0,0,0));
        selectionConnection = new Connection(new Point2D(0,0), new Point2D(0,0), null, null);
        this.boxesOcr = boxesOcr;
        this.boxesCell = boxesCells;
        this.connections = connections;
    }

    public Drawable getSelectionBox() {
        return selectionBox;
    }

    public Drawable getSelectionConnection() {
        return selectionConnection;
    }

    private void dragWithoutConnection(Rectangle2D rectangle, double scale) {
        selectionBox.setBoundingBox(new Rectangle2D(rectangle.getMinX() / scale, rectangle.getMinY() / scale, rectangle.getWidth() / scale, rectangle.getHeight() / scale));
    }

    private void dragWithConnection(Point2D startPoint, Point2D endPoint, double scale) {
        selectionConnection.setPoints(new Point2D(startPoint.getX() / scale, startPoint.getY() / scale), new Point2D(endPoint.getX() / scale, endPoint.getY() / scale));

    }

    @Override
    public void drag(Point2D startPoint, Point2D endPoint, double scale) {
//        if (connectMode) {
//            dragWithConnection(startPoint, endPoint, scale);
//        }
//        else {
//            dragWithoutConnection(new Rectangle2D(Math.min(startPoint.getX(), endPoint.getX()), Math.min(startPoint.getY(), endPoint.getY()), Math.abs(endPoint.getX() - startPoint.getX()), Math.abs(endPoint.getY() - startPoint.getY())), scale);
//        }

        dragWithoutConnection(new Rectangle2D(Math.min(startPoint.getX(), endPoint.getX()), Math.min(startPoint.getY(), endPoint.getY()), Math.abs(endPoint.getX() - startPoint.getX()), Math.abs(endPoint.getY() - startPoint.getY())), scale);
    }

    void select(boolean select, Vector<Box> boxes, double scale, Rectangle2D rectangle) {
        for(Box box : boxes) {
            box.select(rectangle, scale);
        }
    }

    private void dragReleasedWithoutConnection(Rectangle2D rectangle, double scale, MouseButton button) {
        selectionBox.setBoundingBox(new Rectangle2D(0,0,0,0));
        if (button == MouseButton.PRIMARY) {
            // For OCR
            select(true, this.boxesOcr.getBoxes(), scale, rectangle);
            select(true, boxesCell.getBoxes(), scale, new Rectangle2D(0,0,0,0));
        }
        else if (button == MouseButton.SECONDARY) {
            // For Cell boxes
            select(true, boxesCell.getBoxes(), scale, rectangle);
            select(true, boxesOcr.getBoxes(), scale, new Rectangle2D(0,0,0,0));
        }
    }

    private void dragReleasedWithConnection(Rectangle2D rectangle, double scale, MouseButton button) {
        if (button != MouseButton.SECONDARY)
            return;
        select(true, boxesOcr.getBoxes(), scale, new Rectangle2D(0,0,0,0));
        selectionBox.setBoundingBox(new Rectangle2D(0,0,0,0));

        selectionConnection.setPoints(new Point2D(0,0), new Point2D(0,0));

        Vector<Box> boxes = boxesCell.getBoxes();
        Vector<Box> selectedBoxes = new Vector<>();
        for(Box box : boxes) {
            if(box.isSelected())
                selectedBoxes.add(box);
        }

        if (selectedBoxes.size() == 0) {
            // For Cell boxes
            select(true, boxesCell.getBoxes(), scale, rectangle);
            select(true, boxesOcr.getBoxes(), scale, new Rectangle2D(0,0,0,0));
            return;
        }
        else {
            Box willBeSelected = null;
            boolean multipleSelected = false;
            for(Box box : boxes) {
                if(box.willBeSelected(rectangle, scale)) {
                    if (willBeSelected == null) {
                        willBeSelected = box;
                    }
                    else {
                        multipleSelected = true;
                    }
                }
            }
            if (!multipleSelected && willBeSelected != null) {
                Point2D endPoint = new Point2D(willBeSelected.getBoundingBox(scale).getMaxX(), willBeSelected.getBoundingBox(scale).getMaxY());
                for (Box alreadySelectedBox : selectedBoxes) {
                        Point2D startPoint = new Point2D(alreadySelectedBox.getBoundingBox(scale).getMinX(), alreadySelectedBox.getBoundingBox(scale).getMinY());
                    Connection newConnection = new Connection(startPoint, endPoint, ((CellBox)alreadySelectedBox), (CellBox)willBeSelected);
                    connections.getConnections().add(newConnection);
                }
            }
        }

        select(true, boxesCell.getBoxes(), scale, new Rectangle2D(0,0,0,0));
        select(true, boxesOcr.getBoxes(), scale, new Rectangle2D(0,0,0,0));
    }

    @Override
    public void dragReleased(Rectangle2D rectangle, double scale, MouseButton button) {
        if (!connectMode)
            dragReleasedWithoutConnection(rectangle, scale, button);
        else
            dragReleasedWithConnection(rectangle, scale, button);
    }

    @Override
    public void keyPressed(double scale, KeyCode keyCode) {
        if (keyCode == KeyCode.DELETE || keyCode == KeyCode.D) {
            // Delete the selected boxes
            Vector<Box>boxes = boxesOcr.getBoxes();
            Vector<Box> toBeDeleted = new Vector<>();
            for(Box box: boxes) {
                if (box.isSelected())
                    toBeDeleted.add(box);
            }
            for (Box b:toBeDeleted) {
                boxes.remove(b);
            }
            boxes = boxesCell.getBoxes();
            toBeDeleted = new Vector<>();
            for(Box box: boxes) {
                if (box.isSelected())
                    toBeDeleted.add(box);
            }
            for (Box b:toBeDeleted) {
                boxes.remove(b);
            }
        }

        if (keyCode == KeyCode.C) {
            Vector<Box>boxes = boxesOcr.getBoxes();
            if(boxes.size()==0)
                return;
            Rectangle2D union = null;
            for(Box box: boxes) {
                if (box.isSelected()) {
                    union = Utils.union(union, box.getBoundingBox(scale));
                }
            }

            Box cellBox = new CellBox(union);
            boxesCell.getBoxes().add(cellBox);

        }
    }

    void onToggleConnect(boolean isSelected) {
        this.connectMode = isSelected;
    }
}
