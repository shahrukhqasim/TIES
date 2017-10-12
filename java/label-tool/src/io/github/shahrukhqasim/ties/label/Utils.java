package io.github.shahrukhqasim.ties.label;

import java.io.File;
import java.io.FileInputStream;

/**
 * Created by srq on 12.10.17.
 */
public class Utils {

    static String readTextFile(String path) throws Exception{
        File file = new File(path);
        FileInputStream fis = new FileInputStream(file);
        byte[] data = new byte[(int) file.length()];
        fis.read(data);
        fis.close();
        return new String(data, "UTF-8");
    }
}
