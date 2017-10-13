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
    String[] listOfDirectories;
    IOManager(Controller controller) {
        this.controller = controller;
    }

    String imagePath;
    String ocrPath;
    String cellsPath;
    String logicalCellsPath;

    int currentIndex = 0;

    private void loadOcrBoxes() {
        try {
            String path = this.ocrPath;
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
            String path = this.cellsPath;
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
            controller.boxesCells = new Boxes(boxes);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    void initialize() {
        open(true);
        load();
    }

    void load() {
        synchronized (controller.lock) {
            try {
                if (controller.updater != null)
                    controller.updater.cancel();
                controller.scale = 1;
                loadOcrBoxes();
                loadCellBoxes();
                controller.connections = new Connections(new Vector<>());

                controller.interactionManager = new InteractionManager(controller.boxesOcr, controller.boxesCells, controller.connections);
                controller.selectionBox = controller.interactionManager.getSelectionBox();
                controller.selectionConnection = controller.interactionManager.getSelectionConnection();

                controller.image = new RasterImage(SwingFXUtils.toFXImage(ImageIO.read(new File(this.imagePath)), null));
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
                controller.fileLabel.setText(imagePath);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
        controller.onZoomOut();
        controller.onZoomOut();
        controller.onZoomOut();
        controller.onZoomOut();
    }


    void open(boolean haveTo) {
        try {
            synchronized (controller.lock) {
//                FileChooser chooser = new FileChooser();
//                chooser.setTitle("Open File");
                File file = new File("/home/srq/Datasets/tables/unlv/sorted/samples.txt");//chooser.showOpenDialog(new Stage());
                if (file == null) {
                    throw new Exception("Error in opening file");
                }
                String listOfDirectories = Utils.readTextFile(file.getAbsolutePath());
                this.listOfDirectories = listOfDirectories.split("\\r?\\n");
                if (this.listOfDirectories.length == 0) {
                    throw new Exception("Nothing in the file");
                }
                currentIndex = 0;
                this.imagePath = this.listOfDirectories[0] + "/image.png";
                this.cellsPath = this.listOfDirectories[0] + "/cells.json";
                this.ocrPath = this.listOfDirectories[0] + "/ocr.json";
                this.logicalCellsPath = this.listOfDirectories[0] + "/cells_logical.json";
            }
            load();

        }
        catch (Exception e) {
            e.printStackTrace();
            if (haveTo)
                System.exit(-1);
        }
    }

    void toNewIndex(int newIndex) {
        try {
            String dir = this.listOfDirectories[newIndex];
            if (dir.length()==0)
                throw new Exception("Empty");

            this.imagePath = dir + "/image.png";
            this.cellsPath = dir + "/cells.json";
            this.ocrPath = dir + "/ocr.json";
            this.logicalCellsPath = dir + "/cells_logical.json";

            this.currentIndex = newIndex;

        }
        catch (Exception e) {

        }

    }

    void next() {
        toNewIndex(currentIndex + 1);
        load();
    }
    void previous() {
        toNewIndex(currentIndex - 1);
        load();
    }
}
