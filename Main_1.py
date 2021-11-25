from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog,QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton,QApplication, QLabel
from UI_1 import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import main_back
from matplotlib.figure import Figure
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.pyplot as plt
import cv2 as cv
from PyQt5.QtGui import QPixmap, QIcon



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.toolButton.setText('Import')
        self.ui.toolButton.clicked.connect(self.msg)


    def msg(self):
        global fileName
        fileName, filetype = QFileDialog.getOpenFileName(self,"選取檔案", "./","All Files (*);;Text Files (*.txt)")  # 設定副檔名過濾,注意用雙分號間隔
        pixmap = QPixmap(fileName)
        self.ui.label_3.setPixmap(pixmap)
        # self.ui.label_3.resize(pixmap.width()/1.5, pixmap.height()/1.5)





if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
