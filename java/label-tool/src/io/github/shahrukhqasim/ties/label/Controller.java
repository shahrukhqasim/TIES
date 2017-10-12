package io.github.shahrukhqasim.ties.label;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.geometry.BoundingBox;
import javafx.geometry.Rectangle2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

import org.json.*;
import sun.misc.IOUtils;

public class Controller {
    Drawable image;
    Drawable boxesOcr;
    Drawable boxesCells;
    InteractionManager interactionManager;
    Object lock = new Object();
    public ScrollPane scrollPane;
    public Canvas canvas;
    double scale = 1;
    Timer updater;
    public Label zoomLabel;


    public Controller() {
    }

    void loadOcrBoxes() {
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
            Boxes ocrBoxes = new Boxes(boxes);
            this.boxesOcr = ocrBoxes;
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    void initialize() {
        synchronized (lock) {
            try {
                loadOcrBoxes();
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

                        }
                    }
                }, 100, 100);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    void redraw() {
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

    @FXML
    void onCanvasMoved() {

    }
    @FXML
    void onCanvasPressed() {

    }
    @FXML
    void onCanvasReleased() {

    }
    @FXML
    void onCanvasDragged() {

    }
}
