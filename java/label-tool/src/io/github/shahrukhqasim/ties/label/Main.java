package io.github.shahrukhqasim.ties.label;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class Main extends Application {
    Controller controller;

    @Override
    public void start(Stage primaryStage) throws Exception{
        FXMLLoader loader = new FXMLLoader(getClass().getResource("ui_layout.fxml"));
        Parent root = loader.load();
        controller = loader.getController();
        primaryStage.setTitle("TIES - Label Tool");
        primaryStage.setScene(new Scene(root, 1024, 576));
        primaryStage.show();
        controller.initialize();
    }

    @Override
    public void stop() throws Exception {
        controller.onSave();

        super.stop();
        System.exit(0);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
