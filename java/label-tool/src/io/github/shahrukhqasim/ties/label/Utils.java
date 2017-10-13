package io.github.shahrukhqasim.ties.label;

import javafx.geometry.Rectangle2D;

import java.awt.*;
import java.io.File;
import java.io.FileInputStream;

/**
 * Created by srq on 12.10.17.
 */
public class Utils {

    public static String readTextFile(String path) throws Exception {
        File file = new File(path);
        FileInputStream fis = new FileInputStream(file);
        byte[] data = new byte[(int) file.length()];
        fis.read(data);
        fis.close();
        return new String(data, "UTF-8");
    }

    public static Rectangle2D union(Rectangle2D a, Rectangle2D b) {
        if (a == null)
            return b;
        if (b == null)
            return a;
        double x1 = Math.min(a.getMinX(), b.getMinX());
        double y1 = Math.min(a.getMinY(), b.getMinY());
        double x2 = Math.max(a.getMaxX(), b.getMaxX());
        double y2 = Math.max(a.getMaxY(), b.getMaxY());
        return new Rectangle2D(x1, y1, x2 - x1, y2 - y1);
    }

    public static Rectangle2D intersection(Rectangle2D a, Rectangle2D b) {
        Rectangle a2 = new Rectangle((int)a.getMinX(), (int)a.getMinY(), (int)a.getWidth(), (int)a.getHeight());
        Rectangle b2 = new Rectangle((int)b.getMinX(), (int)b.getMinY(), (int)b.getWidth(), (int)b.getHeight());
        Rectangle i = a2.intersection(b2);
        return new Rectangle2D(i.x, i.y, i.width, i.height);
    }

}
