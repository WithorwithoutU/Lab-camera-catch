from PyQt5 import QtWidgets
from designer_window import Ui_Form  # Import your UI class
from PyQt5.QtWidgets import *
import cv2
from PyQt5 import QtCore, QtGui
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import time
import cv2
import imutils
import numpy as np
import math
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pypylon import pylon
from PIL import Image
import os
from sys import exit as sexit
from PyQt5.QtGui import QPixmap
import pyvisa

class MyWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # Create a new layout for cameracatch
        self.layout = QtWidgets.QVBoxLayout(self.cameracatch)
        
        # Create a Label to display camera frames, and add it to the layout
        self.Cam_img = QtWidgets.QLabel()
        self.Cam_img.setScaledContents(True)  # Set the image in the Label to automatically scale to fill the entire Label
        self.Cam_img.setMinimumSize(678, 483)  # Set the minimum size of the Label
        self.layout.addWidget(self.Cam_img)
        # Set the alignment of the layout to center
        self.layout.setAlignment(self.Cam_img, QtCore.Qt.AlignCenter)
        
        # Global variables
        self.points = []
        self.res2 = 0
        self.res1 = 0  # Define res1
        # Initialize camera
        self.cam_img = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cam_img.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # Initialize QTimer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.show_video)
        self.timer.start(1000/30)  # Update frequency is 30 frames per second
        
        # Connect pushButton_3's clicked signal to ScreenCapture function
        self.pushButton_3.clicked.connect(self.ScreenCapture)
        
        # Assign the run() function to the runTest button
        self.runTest.clicked.connect(self.run)
        # Assign the save() function to the pushButton
        self.pushButton.clicked.connect(self.save)
        # Assign the reco() function
        self.comboBox.activated.connect(self.reco)
             
        # Connect the 'toggled' signal of each radio button to a function
        self.RRAM.toggled.connect(self.radioButtonClicked)
        self.DC.toggled.connect(self.radioButtonClicked)
        self.Triangle.toggled.connect(self.radioButtonClicked)
        
        self.pushButton_2.clicked.connect(self.selectPath)
        # For J-V plot
        self.fig1 = Figure(figsize=(5,3), dpi=100)
        self.plot1 = self.fig1.add_subplot()
        canvas1 = FigureCanvas(self.fig1)
        toolbar1 = NavigationToolbar(canvas1, self.frame_7)

        # Create a QVBoxLayout
        layout1 = QtWidgets.QVBoxLayout(self.frame_7)
        layout1.addWidget(canvas1)
        layout1.addWidget(toolbar1)       
        
        # For logJ-V plot
        self.fig2 = Figure(figsize=(5,3), dpi=100)
        self.plot2 = self.fig2.add_subplot()
        canvas2 = FigureCanvas(self.fig2)
        toolbar2 = NavigationToolbar(canvas2, self.frame_5)

        # Create another QVBoxLayout
        layout2 = QtWidgets.QVBoxLayout(self.frame_5)
        layout2.addWidget(canvas2)
        layout2.addWidget(toolbar2)

    def resizeEvent(self, event):
        # When the window size changes, set the Label size to match cameracatch
        self.Cam_img.setGeometry(0, 0, self.cameracatch.width(), self.cameracatch.height())

    def pylon_image_to_cv2(self, grabResult):
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        image = converter.Convert(grabResult)
        img = image.GetArray()
        return img

    # Display the OpenCV image on the Cam_img in the interface
    def show_video(self):
        if self.cam_img.IsGrabbing():
            grabResult = self.cam_img.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                img = self.pylon_image_to_cv2(grabResult)
                self.show_cv_img(img)
            grabResult.Release()

    def show_cv_img(self, img):
        shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = shrink.shape
        bytesPerLine = channel * width
        QtImg = QtGui.QImage(shrink.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.jpg_out = QtGui.QPixmap(QtImg).scaled(self.Cam_img.width(), self.Cam_img.height(), QtCore.Qt.KeepAspectRatio)
        self.Cam_img.setPixmap(self.jpg_out)

    def radioButtonClicked(self):
        if self.RRAM.isChecked():
            self.waveDisplay.setText("RRAM")
        elif self.DC.isChecked():
            self.waveDisplay.setText("DC")
        elif self.Triangle.isChecked():
            self.waveDisplay.setText("Triangle")
        # Set the text alignment to center
        self.waveDisplay.setAlignment(QtCore.Qt.AlignCenter)

    def selectPath(self):
        # Use QFileDialog to open a file selection dialog
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.AnyFile)

        if file_dialog.exec_():
            self.file_path_list = file_dialog.selectedFiles()
            file_name = self.file_path_list[0]
            self.lineEdit_10.setText(file_name)

    def reco(self, _=None):  # Press the calculate button and calculate the diameter
        self.distance = 0
        self.points = []
        self.res2 = 0
        mode = self.comboBox.currentText()  # Get the currently selected text in QComboBox
        img_path = self.lineEdit_10.text()  # Get the text in lineEdit_10 as the image path
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding for better results
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Use morphological operations with appropriate kernel sizes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        th7 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=3)
        th7 = cv2.morphologyEx(th7, cv2.MORPH_OPEN, kernel, iterations=3)
    
        if mode == 'Line':
            # Manual measurement
            res1 = [0]  # Use a list to allow modification inside nested function
            img_line = th7.copy()

            def mouse_callback(event, x, y, flags, userdata):
                if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
                    img_display = img_line.copy()
                    cv2.line(img_display, (0, y), (img_display.shape[1], y), (0, 255, 0), 3)
                    row = th7[y]
                    indices = np.where(row > 0)[0]
                    if len(indices) >= 2:
                        start_pos = indices[0]
                        end_pos = indices[-1]
                        if event == cv2.EVENT_LBUTTONDOWN:
                            res1[0] = end_pos - start_pos
                            print("Measured length:", res1[0])
                    cv2.imshow('res', img_display)

            cv2.namedWindow("res", 0)
            cv2.resizeWindow("res", 1280, 720)
            cv2.setMouseCallback('res', mouse_callback)
            cv2.imshow('res', img_line)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            distance = res1[0] * 0.3

        elif mode == 'Automatically':
            # Find contours in the binary image
            contours_list, hierarchy = cv2.findContours(th7, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Initialize variables
            res = float('inf')
            res_x1 = res_x2 = res_y = 0

            for cnt in contours_list:
                x, y, w, h = cv2.boundingRect(cnt)
                if 100 < w < res:  # Set a minimum width threshold
                    res = w
                    res_x1 = x
                    res_x2 = x + w
                    res_y = y + h // 2  # Use the center of the bounding box vertically

            if res != float('inf'):
                distance = res * 0.3
                cv2.namedWindow("res", 0)
                cv2.resizeWindow("res", 1280, 720)
                img_line = th7.copy()
                cv2.line(img_line, (res_x1, res_y), (res_x2, res_y), (0, 255, 0), 4)
                cv2.imshow('res', img_line)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No suitable contour found.")
                distance = 0

        elif mode == 'Manually':
            # Manual measurement by clicking two points
            res2 = [0]
            img_line = img.copy()

            def mouse_callback(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.points.append((x, y))
                    if len(self.points) == 2:
                        cv2.line(img_line, self.points[0], self.points[1], (0, 255, 0), 4)
                        res2[0] = abs(self.points[1][0] - self.points[0][0])
                        cv2.imshow('res', img_line)
                        print("Measured length:", res2[0])
                        self.points = []

            cv2.namedWindow("res", 0)
            cv2.resizeWindow("res", 1280, 720)
            cv2.setMouseCallback('res', mouse_callback)
            cv2.imshow('res', img_line)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            distance = res2[0] * 0.3

        # Update the content of label_16
        self.label_16.setText(" %i" % distance)

    def run(self, event):  # Select the wave
        am = int(self.lineEdit.text())
        points = int(self.lineEdit_3.text())
        cycles = int(self.lineEdit_2.text())
        mode = self.waveDisplay.text()
        # Get text
        text = self.label_16.text()
        # Convert to integer
        dis = int(text)
        rm = pyvisa.ResourceManager()  # Configure the 2612
        print(dis)
        address = "GPIB0::26::INSTR"
        inst = rm.open_resource(address)
        inst.timeout = 1000000  
        inst.write("smub.sense = smub.SENSE_REMOTE")
        inst.write("smub.measure.autorangei = smub.AUTORANGE_ON")
        inst.write("smub.measure.nplc = 1")
        inst.write("smub.nvbuffer1.clear()")
        inst.write("smub.nvbuffer2.clear()")
        global voltage
        global abscurrent
        global logcurrent
        voltage = []
        current = []
        abscurrent = []
        logcurrent = [] 
        record_p = []
        record_n = []
        current1 = []
        if mode == 'Triangle':
            high = float(am)
            low = float(am) * -1
            for times in range(cycles):
                inst.write("SweepVLinMeasureI(smub, %f, %f, 1e-3, %f )" % (high, low, points))
                for i in range(1, points + 1):
                    voltage.append(float(inst.query("print(smub.nvbuffer1.sourcevalues[%f])" % i)))
                    current.append(float(inst.query("print(smub.nvbuffer1[%f])" % i)))
                inst.write("SweepVLinMeasureI(smub, %f, %f, 1e-3, %f )" % (low, high, points))
                for i in range(1, points + 1):
                    voltage.append(float(inst.query("print(smub.nvbuffer1.sourcevalues[%f])" % i)))
                    current.append(float(inst.query("print(smub.nvbuffer1[%f])" % i)))

        elif mode == 'DC':
            high = float(am)
            for times in range(cycles):
                inst.write("SweepVLinMeasureI(smub, %f, %f, 1e-3, %f )" % (high, high, points))
                for i in range(1, points + 1):
                    voltage.append(float(inst.query("print(smub.nvbuffer1.sourcevalues[%f])" % i)))
                    current.append(float(inst.query("print(smub.nvbuffer1[%f])" % i)))
                dictionary = dict(zip(voltage, current))
                postive = abs(dictionary.get(high))
                record_p.append(postive)

        elif mode == 'RRAM':
            high = float(am)
            low = float(am) * -1
            for times in range(cycles):
                inst.write("SweepVLinMeasureI(smub, 0, %f, 1e-3, %f )" % (high, points))  # 0 to -1
                for i in range(1, points + 1):
                    voltage.append(float(inst.query("print(smub.nvbuffer1.sourcevalues[%f])" % i)))  # -1 
                    current.append(float(inst.query("print(smub.nvbuffer1[%f])" % i)))
                dictionary = dict(zip(voltage, current))
                try:
                    negative = abs(dictionary.get(high, 1))
                    record_n.append(negative)
                except KeyError:
                    print("Key not found in dictionary: ", high)

                inst.write("SweepVLinMeasureI(smub, %f, 0, 1e-3, %f )" % (high, points))  # -1 to 0
                for i in range(1, points + 1):
                    voltage.append(float(inst.query("print(smub.nvbuffer1.sourcevalues[%f])" % i)))
                    current.append(float(inst.query("print(smub.nvbuffer1[%f])" % i)))
                dictionary = dict(zip(voltage, current))
                try:
                    negative = abs(dictionary.get(high, 1))
                    record_n.append(negative)
                except KeyError:
                    print("Key not found in dictionary: ", high)

                inst.write("SweepVLinMeasureI(smub, 0, %f, 1e-3, %f )" % (low, points))  # 0 to 1
                for i in range(1, points + 1):
                    voltage.append(float(inst.query("print(smub.nvbuffer1.sourcevalues[%f])" % i)))
                    current.append(float(inst.query("print(smub.nvbuffer1[%f])" % i)))
                dictionary = dict(zip(voltage, current))
                try:
                    positive = abs(dictionary.get(low, -1))
                    record_p.append(positive)
                except KeyError:
                    print("Key not found in dictionary: ", low)
                    
                inst.write("SweepVLinMeasureI(smub, %f, 0, 1e-3, %f )" % (low, points))  # 1 to 0
                for i in range(1, points + 1):
                    voltage.append(float(inst.query("print(smub.nvbuffer1.sourcevalues[%f])" % i)))
                    current.append(float(inst.query("print(smub.nvbuffer1[%f])" % i)))
                dictionary = dict(zip(voltage, current))
                try:
                    positive = abs(dictionary.get(low, -1))
                    record_p.append(positive)
                except KeyError:
                    print("Key not found in dictionary: ", low)
                       
        area = math.pi * ((dis / 2 * pow(10, -4)) ** 2)  # Calculate the current density
        print(area)
        print(dis)
        for i in range(len(voltage)):
            abscurrent.append(abs(current[i] / area))
            logcurrent.append(np.log10(abscurrent[i]))
            current1.append(abs(current[i]))
           
        # Use self.plot1 and self.plot2 instead of plot1 and plot2
        colors1 = '#DC143C'  # Draw the J-V scatter    
        plot_area = np.pi * 2 ** 2
        self.plot1.scatter(voltage, current1, s=plot_area, c=colors1, alpha=0.4)
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()

        colors2 = '#00CED1'  # Draw the logJ-V scatter    
        self.plot2.scatter(voltage, logcurrent, s=plot_area, c=colors2, alpha=0.4)
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
        
    def save(self, event):
        outputfile = str(self.lineEdit_5.text())
        outputlog = str(self.lineEdit_4.text())
        outputfile = str(outputfile + '.xlsx')
        outputlog = str(outputlog + '.xlsx')

        # Integrate the data
        terJ = np.column_stack((voltage, abscurrent))
        terJ1 = np.column_stack((voltage, logcurrent))

        df = pd.DataFrame(terJ, columns=['Voltage', 'J'])
        df.to_excel(outputfile, index=False)

        df1 = pd.DataFrame(terJ1, columns=['Voltage', 'logJ'])
        df1.to_excel(outputlog, index=False)
        
    def qimage_to_pil_image(self, qimage):
        buffer = qimage.bits().asstring(qimage.byteCount())
        img = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "RGBA", 0, 1)
        return img.convert("RGB")
            
    def ScreenCapture(self):
        name = "demo.jpg"
        save_path = os.path.join("D:\\Test\\image\\") + name
        
        # Get the QPixmap object from the Label
        pixmap = self.Cam_img.pixmap()
        
        # Convert the QPixmap object to QImage
        qimage = pixmap.toImage()
        
        # Convert the QImage object to PIL image
        img = self.qimage_to_pil_image(qimage)
        
        # Resize the image
        img = img.resize((3088, 2064), Image.ANTIALIAS)
        
        # Save the image
        img.save(save_path)
                
    def qimage_to_opencv(self, qimage):
        # This function converts a QImage object to an OpenCV image
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr

    def exitprogram(self):
        cv2.destroyAllWindows()
        self.close()
        sexit()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
