package io.github.shahrukhqasim.ties.label;

import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Rectangle2D;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.json.JSONArray;
import org.json.JSONObject;

import javax.imageio.ImageIO;
import java.io.File;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

/**
 * Created by srq on 13.10.17.
 */
public class IOManager {
    Controller controller;
    List<String> listOfDirectories;
    IOManager(Controller controller) {
        this.controller = controller;
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
            controller.boxesOcr = new Boxes(boxes);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void loadCellBoxes() {
        if (controller.boxesOcr == null) {
            System.err.println("Error loaded cell boxes");
        }

        try {
            String path = "/home/srq/Datasets/tables/unlv/sorted/0101_003-0/cells.json";
            String text = Utils.readTextFile(path);

            JSONObject json = new JSONObject(text);
            JSONArray array = json.getJSONArray("cells");

            Vector<Box> ocrBoxes = controller.boxesOcr.getBoxes();

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
                    Rectangle2D ocrRectangle = ocrBoxes.get(j).getBoundingBox(controller.scale);
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
            controller.boxesCells = new Boxes(boxes);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    void initialize() {
        synchronized (controller.lock) {
            try {
                if (controller.updater != null)
                    controller.updater.cancel();
                controller.scale = 1;
                loadOcrBoxes();
                loadCellBoxes();

                controller.interactionManager = new InteractionManager(controller.boxesOcr, controller.boxesCells);
                controller.selectionBox = controller.interactionManager.getSelectionBox();

                controller.image = new RasterImage(SwingFXUtils.toFXImage(ImageIO.read(new File("/home/srq/Datasets/tables/unlv/sorted/0101_003-0/image.png")), null));
                controller.canvas.setWidth(controller.image.getBoundingBox(controller.scale).getWidth());
                controller.canvas.setHeight(controller.image.getBoundingBox(controller.scale).getHeight());

                controller.updater = new Timer();
                controller.updater.scheduleAtFixedRate(new TimerTask() {
                    @Override
                    public void run() {
                        try {
                            controller.redraw();
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


    void open() {
        try {
            FileChooser chooser = new FileChooser();
            chooser.setTitle("Open File");
            File file = chooser.showOpenDialog(new Stage());
            if (file == null)
                System.exit(-1);
            String listOfDirectories = Utils.readTextFile(file.getAbsolutePath());
            String lines[] = listOfDirectories.split("\\r?\\n");
            System.out.print(lines.length);

        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }
    void next() {

    }
    void previous() {

    }
}
