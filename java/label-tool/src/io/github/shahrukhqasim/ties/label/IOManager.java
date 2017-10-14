package io.github.shahrukhqasim.ties.label;

import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Point2D;
import javafx.geometry.Rectangle2D;
import org.json.JSONArray;
import org.json.JSONObject;

import javax.imageio.ImageIO;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;

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

    /**
     * Loads cell boxes into memory. If connections are also loaded, returns true otherwise false
     *
     * Tries to find logical cell boxes with the help of OCR boxes and proposed structured cell boxes. However, if the
     * file for logical cell boxes already exists, it just picks logical cell boxes and connections from there.
     *
     * @return true if it also loads connections
     */
    private boolean loadCellBoxes() {
        if (controller.boxesOcr == null) {
            System.err.println("Error loading cell boxes");
        }

        try {
            // If logical cells file already exists, just pick the cells and connections from there
            File logicalCellsFile = new File(this.logicalCellsPath);
            HashMap<Integer, CellBox> mapOfCellBoxes = new HashMap<>();
            if (logicalCellsFile.exists()) {
                String text = Utils.readTextFile(this.logicalCellsPath);
                JSONObject json = new JSONObject(text);
                JSONArray array = json.getJSONArray("cells");
                Vector<Box> boxes = new Vector<>();
                for (int i = 0; i < array.length(); i++) {
                    JSONObject object = array.getJSONObject(i);
                    int x1 = object.getInt("x1");
                    int y1 = object.getInt("y1");
                    int width = object.getInt("x2") - x1;
                    int height = object.getInt("y2") - y1;
                    int id = object.getInt("id");
                    Rectangle2D rectangle2D = new Rectangle2D(x1, y1, width, height);
                    CellBox cellBox = new CellBox(rectangle2D, id);
                    boxes.add(cellBox);
                    mapOfCellBoxes.put(id, cellBox);
                }
                array = json.getJSONArray("connections");
                Vector<Connection>connections = new Vector<>();
                for (int i = 0; i < array.length(); i++) {
                    JSONObject object = array.getJSONObject(i);
                    int childNode = object.getInt("child_node");
                    int parentNode = object.getInt("parent_node");
                    CellBox nodeA = mapOfCellBoxes.get(childNode);
                    CellBox nodeB = mapOfCellBoxes.get(parentNode);
                    Point2D endPoint = new Point2D(nodeB.getBoundingBox(1).getMaxX(), nodeB.getBoundingBox(1).getMaxY());
                    Point2D startPoint = new Point2D(nodeA.getBoundingBox(1).getMinX(), nodeA.getBoundingBox(1).getMinY());
                    Connection connection = new Connection(startPoint, endPoint, nodeA, nodeB);
                    connections.add(connection);
                }

                controller.boxesCells = new Boxes(boxes);
                controller.connections = new Connections(connections);
                return true;
            }
            else {
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
                            if (areaIntersection > 0.9 * areaOriginal) {
                                intersectionCriteriaMet = true;
                            }
                        }
                        if (intersectionCriteriaMet) {
                            if (innerRect == null) {
                                innerRect = intersectionRect;
                            } else {
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
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    void initialize() {
        open(true);
        load();
    }

    void load() {
        if (controller.lock.tryLock()) {
            try {
                if (controller.updater != null)
                    controller.updater.cancel();
                controller.scale = 1;
                loadOcrBoxes();
                if (!loadCellBoxes())
                    controller.connections = new Connections(new Vector<>());

                controller.interactionManager = new InteractionManager(controller.boxesOcr, controller.boxesCells, controller.connections);
                controller.interactionManager.onToggleConnect(controller.toggleButtonConnect.isSelected());
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
            finally {
                controller.lock.unlock();
            }
        }
        controller.onZoomOut();
        controller.onZoomOut();
        controller.onZoomOut();
        controller.onZoomOut();
    }


    void open(boolean haveTo) {
        try {
             if (controller.lock.tryLock()) {
                 try {
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
                 finally {
                     controller.lock.unlock();
                 }

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
        if (controller.toggleButtonAutoSave.isSelected())
            save();

        toNewIndex(currentIndex + 1);
        load();
    }
    void previous() {
        if (controller.toggleButtonAutoSave.isSelected())
            save();

        toNewIndex(currentIndex - 1);
        load();
    }
    void save() {
        try {
            JSONObject jsonComplete = new JSONObject();
            JSONArray jsonCellsArray = new JSONArray();
            // Iterate through all the cells
            Vector<Box> cellBoxes = controller.boxesCells.getBoxes();
            for (Box b : cellBoxes) {
                JSONObject jsonCellObject = new JSONObject();
                CellBox cellBox = (CellBox)b;
                jsonCellObject.put("id", cellBox.getId());
                Rectangle2D boundingRect = cellBox.getBoundingBox(1);
                jsonCellObject.put("x1", boundingRect.getMinX());
                jsonCellObject.put("y1", boundingRect.getMinY());
                jsonCellObject.put("x2", boundingRect.getMaxX());
                jsonCellObject.put("y2", boundingRect.getMaxY());
                jsonCellsArray.put(jsonCellObject);
            }
            // Iterate through connections
            Vector<Connection> connections = controller.connections.getConnections();
            JSONArray jsonConnectionsArray = new JSONArray();
            for (Connection c : connections) {
                JSONObject jsonConnectionObject = new JSONObject();
                jsonConnectionObject.put("child_node", c.getNodeA().getId());
                jsonConnectionObject.put("parent_node", c.getNodeB().getId());
                jsonConnectionsArray.put(jsonConnectionObject);
            }

            // Add cells and connections to complete json object
            jsonComplete.put("cells", jsonCellsArray);
            jsonComplete.put("connections", jsonConnectionsArray);

            // Write to the output file
            String encodedJson = jsonComplete.toString();
            BufferedWriter writer = new BufferedWriter(new FileWriter(logicalCellsPath));
            writer.write (encodedJson);
            writer.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
