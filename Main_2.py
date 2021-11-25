from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog,QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton,QApplication, QLabel
from UI2 import Ui_Form
import cv2 as cv
from PyQt5.QtGui import QPixmap, QIcon, QColor
import DFT
import reconstruction
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

file = False
state = False

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.lineEdit.setText('y = -1~1')
        # Proces Message
        self.ui.label_5.setText('System Message')
        self.ui.plainTextEdit.setStyleSheet(
                            """QPlainTextEdit {background-color: #000000;
                           color: #FFFAF0;
                           font-family: Arial;}""")


        # Import File/Image
        self.ui.toolButton.setText('Import')
        self.ui.toolButton.clicked.connect(self.msg)

        # FT
        self.ui.label_4.setText('2D Fourier Transform')
        self.ui.pushButton_2.setText('Show')
        self.ui.pushButton_2.clicked.connect(self.FT)


        # Radon Transform
        self.ui.label_6.setText('Radon Transform')
        self.ui.pushButton_3.setText('Show')
        self.ui.pushButton_3.clicked.connect(self.randon)

        # self.ui.pushButton_2.clicked.connect(self.FT)

        # CheckBox
        self.ui.checkBox.setText('Filter or not')
        self.ui.checkBox.stateChanged.connect(self.change_state)

        # Fourier slice + filter back
        self.ui.label_7.setText('Reconstruction')
        self.ui.pushButton_4.setText('Show')
        self.ui.pushButton_4.clicked.connect(self.Reconstruction)

        # Comparision
        self.ui.label_8.setText('Comparision')
        self.ui.pushButton_5.setText('Show')
        self.ui.pushButton_5.clicked.connect(self.compare)

    def show_message(self,msg):
        self.ui.plainTextEdit.appendPlainText(msg)
        # self.ui.plainTextEdit.append(msg)

    def msg(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        global file_path
        file_path = name[0]
        pixmap = QPixmap(name[0])
        img_width = pixmap.width()
        img_height = pixmap.height()
        # print(img_height,img_width)
        if img_width >  460 or img_height > 600:
            scale = max(img_height/460,img_width/600)
            pixmap = pixmap.scaled(img_width/scale,img_height/scale)
        elif img_width <  460 or img_height < 600:
            scale = min(460 / img_height, 600 / img_width)
            pixmap = pixmap.scaled(img_width*scale,img_height*scale)
        if file_path != None:
            global file
            file = True
        print(file)
        self.ui.label_3.setPixmap(pixmap)
        self.show_message('Import '+file_path)


    def FT(self):
        if file:
            try:
                self.show_message("Fourier Transfoem...")
                start = time.time()
                name = DFT.input(file_path=file_path)
                totol_time = time.time()-start
                self.show_message("Processing Time:" + str(totol_time))
                self.show_message("Succesully Complete Fourier Transfoem")
                pixmap = QPixmap(name)
                img_width = pixmap.width()
                img_height = pixmap.height()
                print(img_height,img_width)
                if img_width > 460 or img_height > 600:
                    scale = max(img_height / 460, img_width / 600)
                    pixmap = pixmap.scaled(img_width / scale, img_height / scale)
                elif img_width < 460 or img_height < 600:
                    scale = min(460 / img_height, 600 / img_width)
                    pixmap = pixmap.scaled(img_width * scale, img_height * scale)
                self.ui.label_9.setPixmap(pixmap)
            except Exception as e:
                self.show_message(e)
        else:
            self.show_message("Please select an image")

    def randon(self):
        if file:
            try:
                self.show_message("Radon Transfoem...")
                start = time.time()
                # myImg = dummyImg(500,700)
                myImg = Image.open(file_path).convert('L')  # gray level
                myImgPad, c0, c1 = reconstruction.padImage(myImg)  # PIL image object
                # resolution
                dTheta = 1
                theta = np.arange(0, 181, dTheta)
                print('Getting projections\n')
                sinogram = reconstruction.getProj(myImgPad, theta)  # numpy array
                sinogram = Image.fromarray((sinogram - np.min(sinogram)) / np.ptp(sinogram) * 255)
                totol_time = time.time() - start
                self.show_message("Processing Time:" + str(totol_time))
                self.show_message("Succesully Complete Radon Transfoem")
                pixmap = QPixmap("sinogram.jpg")
                img_width = pixmap.width()
                img_height = pixmap.height()
                # print(img_height,img_width)
                if img_width > 460 or img_height > 600:
                    scale = max(img_height / 460, img_width / 600)
                    pixmap = pixmap.scaled(img_width / scale, img_height / scale)
                elif img_width < 460 or img_height < 600:
                    scale = min(460 / img_height, 600 / img_width)
                    pixmap = pixmap.scaled(img_width * scale, img_height * scale)
                self.ui.label_10.setPixmap(pixmap)
            except Exception as e:
                self.show_message(e)
                print(e)
        else:
            self.show_message("Please select an image")

    def change_state(self):
        global state
        state = not state
        print(state)

    def Reconstruction(self):
        if file:
            try:
                # myImg = dummyImg(500,700)
                self.show_message("Filter: " + str(state))
                self.show_message("Reconstruction...")
                start = time.time()
                csv_path = 'radon_transform.csv'
                re_img = reconstruction.show_re(file_path,csv_path,state)
                totol_time = time.time() - start
                self.show_message("Processing Time:" + str(totol_time))
                self.show_message("Succesully Complete Reconstruction")
                pixmap = QPixmap("re_img.jpg")
                img_width = pixmap.width()
                img_height = pixmap.height()
                # print(img_height,img_width)
                if img_width > 460 or img_height > 600:
                    scale = max(img_height / 460, img_width / 600)
                    pixmap = pixmap.scaled(img_width / scale, img_height / scale)
                elif img_width < 460 or img_height < 600:
                    scale = min(460 / img_height, 600 / img_width)
                    pixmap = pixmap.scaled(img_width * scale, img_height * scale)
                self.ui.label_11.setPixmap(pixmap)
            except Exception as e:
                self.show_message(e)
                print(e)
        else:
            self.show_message("Please select an image")

    def compare(self):
        if file:
            try:
                position = self.ui.lineEdit.text()
                self.show_message("Comparing original image with reconstructed image..." )
                print(file_path)
                original = cv.imread(file_path)
                # original = cv.cvtColor(original, cv.COLOR_BGR2RGB)
                original = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
                print(np.shape(original))
                n = np.shape(original)[1]
                recon = plt.imread("re_img.jpg")
                print(np.shape(recon))
                y = float(position)*n/2 + 100
                print(y)
                print(np.shape(original[:, int(y)]))
                print(np.shape(recon[:, int(y)]))

                plt.plot(original[:, int(y)]*255, label='Original Image')
                plt.plot(recon[:, int(y)], label='Reconstruction')

                plt.legend(fontsize=12)
                plt.title('Position:' + position)
                plt.xlabel("Projection axis")
                plt.ylabel("Intensity")
                plt.savefig("compare.jpg")
                pixmap = QPixmap("compare.jpg")
                img_width = pixmap.width()
                img_height = pixmap.height()
                # print(img_height,img_width)
                if img_width > 460 or img_height > 600:
                    scale = max(img_height / 460, img_width / 600)
                    pixmap = pixmap.scaled(img_width / scale, img_height / scale)
                elif img_width < 460 or img_height < 600:
                    scale = min(460 / img_height, 600 / img_width)
                    pixmap = pixmap.scaled(img_width * scale, img_height * scale)
                self.ui.label_12.setPixmap(pixmap)
                plt.show()
            except Exception as e:
                self.show_message(e)
                print(e)
        else:
            self.show_message("Please select an image")





if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
