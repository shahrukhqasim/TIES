package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;

import java.util.Vector;

/**
 * Created by srq on 12.10.17.
 */
public class InteractionManager implements InteractionListener {
    SelectionBox selectionBox;
    Boxes boxesOcr;
    Boxes boxesCell;
    public InteractionManager(Boxes boxesOcr, Boxes boxesCells) {
        selectionBox = new SelectionBox(new Rectangle2D(0,0,0,0));
        this.boxesOcr = boxesOcr;
        this.boxesCell = boxesCells;
    }

    public Drawable getSelectionBox() {
        return selectionBox;
    }

    @Override
    public void drag(Rectangle2D rectangle, double scale) {
        selectionBox.setBoundingBox(new Rectangle2D(rectangle.getMinX() / scale, rectangle.getMinY() / scale, rectangle.getWidth() / scale, rectangle.getHeight() / scale));
    }

    void select(boolean select, Vector<Box> boxes, double scale, Rectangle2D rectangle) {
        for(Box box : boxes) {
            box.select(rectangle, scale);
        }
    }

    @Override
    public void dragReleased(Rectangle2D rectangle, double scale, MouseButton button) {
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

    @Override
    public void keyPressed(double scale, KeyCode keyCode) {
        if (keyCode == KeyCode.DELETE) {
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
}
