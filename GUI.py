import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QPixmap, QImage
from tensorflow.keras.models import load_model

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.super_resolution_model = load_model('./model/resolution1.h5')
        self.colorization_model = load_model('./model/lab_model.h5')
        self.image_path = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Processor')
        self.setGeometry(300, 300, 400, 300)

        layout = QVBoxLayout()

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        button_open = QPushButton('Open Image')
        button_open.clicked.connect(self.open_image)
        layout.addWidget(button_open)

        button_super_resolution = QPushButton('Apply Super Resolution')
        button_super_resolution.clicked.connect(self.apply_super_resolution)
        layout.addWidget(button_super_resolution)

        button_colorization = QPushButton('Apply Colorization')
        button_colorization.clicked.connect(self.apply_colorization)
        layout.addWidget(button_colorization)

        button_save = QPushButton('Save Processed Image')
        button_save.clicked.connect(self.save_image)
        layout.addWidget(button_save)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(200, 200))

    def apply_super_resolution(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            img = cv2.resize(img, (150, 150))
            x = np.expand_dims(img, axis=0)

            super_resolution_result = self.super_resolution_model.predict(x)

            self.processed_image = np.clip(super_resolution_result[0], 0, 255).astype(np.uint8)

            qimage = QImage(self.processed_image, self.processed_image.shape[1], self.processed_image.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(600, 600))

    def apply_colorization(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            img = cv2.resize(img, (150, 150))
            x = np.expand_dims(img, axis=0)

            colorization_result = self.colorization_model.predict(x)

            self.processed_image = np.clip(colorization_result[0], 0, 255).astype(np.uint8)

            qimage = QImage(self.processed_image, self.processed_image.shape[1], self.processed_image.shape[0],
                            QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(200, 200))

    def save_image(self):
        if hasattr(self, 'processed_image'):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                       "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
            if file_name:
                cv2.imwrite(file_name, self.processed_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    image_processor = ImageProcessor()
    image_processor.show()
    sys.exit(app.exec_())


