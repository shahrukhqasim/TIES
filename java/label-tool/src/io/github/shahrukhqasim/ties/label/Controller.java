package io.github.shahrukhqasim.ties.label;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import org.json.*;

public class Controller {
    private Drawable image;
    private Boxes boxesOcr;
    private Boxes boxesCells;
    private Drawable selectionBox;
    private InteractionManager interactionManager;
    private final Object lock = new Object();
    public ScrollPane scrollPane;
    public Canvas canvas;
    private double scale = 1;
    private Timer updater;
    public Label zoomLabel;


    public Controller() {
    }

    private void loadOcrBoxes() {
        try {
            String path = "/home/srq/Datasets/tables/unlv/sorted/0101_003-0/ocr.json";
            String text = Utils.readTextFile(path);

            JSONObject json = new JSONObject(text);
            JSONArray array = json.getJSONArray("words");

            Vector<Box> boxes = new Vector<>();
            for (int i = 0; i < array.length(); i++) {
                JSONObject object = array.getJSONObject(i);
                int x1 = object.getInt("x1");
                int y1 = object.getInt("y1");
                int width = object.getInt("x2") - x1;
                int height = object.getInt("y2") - y1;
                Rectangle2D rectangle2D = new Rectangle2D(x1, y1, width, height);
                OcrBox box = new OcrBox(rectangle2D);
                boxes.add(box);
            }
            this.boxesOcr = new Boxes(boxes);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void loadCellBoxes() {
        if (boxesOcr == null) {
            System.err.println("Error loaded cell boxes");
        }

        try {
            String path = "/home/srq/Datasets/tables/unlv/sorted/0101_003-0/cells.json";
            String text = Utils.readTextFile(path);

            JSONObject json = new JSONObject(text);
            JSONArray array = json.getJSONArray("cells");

            Vector<Box> ocrBoxes = boxesOcr.getBoxes();

            Vector<Box> boxes = new Vector<>();
            for (int i = 0; i < array.length(); i++) {
                JSONObject object = array.getJSONObject(i);
                int x1 = object.getInt("x1");
                int y1 = object.getInt("y1");
                int width = object.getInt("x2") - x1;
                int height = object.getInt("y2") - y1;
                Rectangle2D rectangle2D = new Rectangle2D(x1, y1, width, height);

                Rectangle2D innerRect = null;

                for (int j = 0; j < ocrBoxes.size(); j++) {
                    Rectangle2D ocrRectangle = ocrBoxes.get(j).getBoundingBox(scale);
                    boolean intersectionCriteriaMet = false;
                    Rectangle2D intersectionRect = null;
                    if (ocrRectangle.intersects(rectangle2D)) {
                        intersectionRect = Utils.intersection(rectangle2D, ocrRectangle);
                        double areaIntersection = intersectionRect.getWidth() * intersectionRect.getHeight();
                        double areaOriginal = ocrRectangle.getWidth() * ocrRectangle.getHeight();
                        if (areaIntersection > 0.9*areaOriginal) {
                            intersectionCriteriaMet = true;
                        }
                    }
                    if (intersectionCriteriaMet) {
                        if (innerRect == null) {
                            innerRect = intersectionRect;
                        }
                        else {
                            innerRect = Utils.union(innerRect, intersectionRect);
                        }
                    }
                }

                if (innerRect != null) {
                    CellBox box = new CellBox(innerRect);
                    boxes.add(box);
                }
            }
            System.out.println("Boxes of cells size is " + boxes.size());
            this.boxesCells = new Boxes(boxes);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    void initialize() {
        synchronized (lock) {
            try {
                loadOcrBoxes();
                loadCellBoxes();

                this.interactionManager = new InteractionManager(this.boxesOcr, this.boxesCells);
                this.selectionBox = this.interactionManager.getSelectionBox();

                image = new RasterImage(SwingFXUtils.toFXImage(ImageIO.read(new File("/home/srq/Datasets/tables/unlv/sorted/0101_003-0/0101_003.png")), null));
                canvas.setWidth(image.getBoundingBox(scale).getWidth());
                canvas.setHeight(image.getBoundingBox(scale).getHeight());

                updater = new Timer();
                updater.scheduleAtFixedRate(new TimerTask() {
                    @Override
                    public void run() {
                        try {
                            redraw();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }, 100, 100);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void redraw() {
        synchronized (lock) {
            double vValue = scrollPane.getVvalue();
            double hValue = scrollPane.getHvalue();

            Rectangle2D boundingBox = image.getBoundingBox(scale);

            double oneWidth = Math.min(scrollPane.getWidth(), boundingBox.getWidth());
            double oneHeight = Math.min(scrollPane.getHeight(), boundingBox.getHeight());

            double x1 = hValue * (boundingBox.getWidth() - oneWidth);
            double y1 = vValue * (boundingBox.getHeight() - oneHeight);
            double x2 = x1 + oneWidth;
            double y2 = y1 + oneHeight;

            GraphicsContext graphics2D = canvas.getGraphicsContext2D();

            Rectangle2D visibleRect = new Rectangle2D(x1, y1, x2 - x1, y2 - y1);

            image.draw(graphics2D, visibleRect, scale);
            boxesOcr.draw(graphics2D, visibleRect, scale);
            boxesCells.draw(graphics2D, visibleRect, scale);
            selectionBox.draw(graphics2D, visibleRect, scale);
        }
    }


    @FXML
    void onZoomIn() {
        synchronized (lock) {
            System.out.println("Zooming in!");
            if (scale == 2)
                return;
            scale += 0.25;
            zoomLabel.setText("Zoom: " + scale + "x");
            Rectangle2D box = image.getBoundingBox(scale);
            canvas.setWidth(box.getWidth());
            canvas.setHeight(box.getHeight());
        }
    }

    @FXML
    void onZoomOut() {
        synchronized (lock) {
            System.out.println("Zooming out!");
            if (scale == 0.25)
                return;
            scale -= 0.25;
            zoomLabel.setText("Zoom: " + scale + "x");
            Rectangle2D box = image.getBoundingBox(scale);
            canvas.setWidth(box.getWidth());
            canvas.setHeight(box.getHeight());
        }
    }

    private Point2D startPointClick;

    @FXML
    void onCanvasMoved() {

    }
    @FXML
    void onCanvasPressed(MouseEvent event) {
        this.startPointClick = new Point2D(event.getX(), event.getY());
    }
    @FXML
    void onCanvasReleased(MouseEvent event) {
        synchronized (lock) {
            synchronized (lock) {
                Point2D endPoint = new Point2D(event.getX(), event.getY());
                interactionManager.dragReleased(new Rectangle2D(Math.min(startPointClick.getX(), endPoint.getX()), Math.min(startPointClick.getY(), endPoint.getY()), Math.abs(endPoint.getX() - startPointClick.getX()), Math.abs(endPoint.getY() - startPointClick.getY())), scale, event.getButton());
            }
        }

    }
    @FXML
    void onCanvasDragged(MouseEvent event) {
        synchronized (lock) {
            Point2D endPoint = new Point2D(event.getX(), event.getY());
            interactionManager.drag(new Rectangle2D(Math.min(startPointClick.getX(), endPoint.getX()), Math.min(startPointClick.getY(), endPoint.getY()), Math.abs(endPoint.getX() - startPointClick.getX()), Math.abs(endPoint.getY() - startPointClick.getY())), scale);
        }
    }

    @FXML
    void onKeyPressed(KeyEvent event) {
        synchronized (lock) {
            interactionManager.keyPressed(scale, event.getCode());
        }
    }
}
