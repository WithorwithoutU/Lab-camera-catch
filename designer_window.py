# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Shitmount")
        Form.resize(1234, 837)
        
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(Form)
      
        self.frame.setLineWidth(-3)
        self.frame.setObjectName("frame")
              
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.input = QtWidgets.QFrame(self.frame)
        self.input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.input.setObjectName("input")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.input)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_10 = QtWidgets.QFrame(self.input)
        self.frame_10.setStyleSheet("border-image: url(:/imgs/logo.gif);")
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_9.addWidget(self.frame_10)
        self.goodLuck = QtWidgets.QLabel(self.input)
        self.goodLuck.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(36, 206, 255, 255), stop:1 rgba(255, 255, 255, 255));")
        self.goodLuck.setAlignment(QtCore.Qt.AlignCenter)
        self.goodLuck.setObjectName("goodLuck")
        self.verticalLayout_9.addWidget(self.goodLuck)
        self.goodLuck_2 = QtWidgets.QLabel(self.input)
        self.goodLuck_2.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(36, 206, 255, 255), stop:1 rgba(255, 255, 255, 255));")
        self.goodLuck_2.setAlignment(QtCore.Qt.AlignCenter)
        self.goodLuck_2.setObjectName("goodLuck_2")
        self.verticalLayout_9.addWidget(self.goodLuck_2)
        self.frame1 = QtWidgets.QFrame(self.input)
        self.frame1.setStyleSheet("font: 75 14pt \"Times New Roman\";\n"
"")
        self.frame1.setObjectName("frame1")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.RRAM = QtWidgets.QRadioButton(self.frame1)
        self.RRAM.setObjectName("RRAM")
        self.horizontalLayout_6.addWidget(self.RRAM)
        self.DC = QtWidgets.QRadioButton(self.frame1)
        self.DC.setObjectName("DC")
        self.horizontalLayout_6.addWidget(self.DC)
        self.Triangle = QtWidgets.QRadioButton(self.frame1)
        self.Triangle.setObjectName("Triangle")
        self.horizontalLayout_6.addWidget(self.Triangle)
        self.verticalLayout_9.addWidget(self.frame1)
        self.waveDisplay = QtWidgets.QLabel(self.input)
        self.waveDisplay.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 189, 167, 250), stop:1 rgba(255, 255, 255, 255));")
        self.waveDisplay.setText("")
        self.waveDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.waveDisplay.setObjectName("waveDisplay")
        self.verticalLayout_9.addWidget(self.waveDisplay)
        self.waveParameters = QtWidgets.QFrame(self.input)
        self.waveParameters.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.waveParameters.setFrameShadow(QtWidgets.QFrame.Raised)
        self.waveParameters.setObjectName("waveParameters")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.waveParameters)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.waveParameters)
        self.groupBox_2.setStyleSheet("font: 75 14pt \"Times New Roman\";\n"
"")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.verticalLayout_4.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.frame2 = QtWidgets.QFrame(self.waveParameters)
        self.frame2.setObjectName("frame2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.lineEdit = QtWidgets.QLineEdit(self.frame2)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_3.addWidget(self.lineEdit)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.frame2)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.verticalLayout_3.addWidget(self.lineEdit_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame2)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_3.addWidget(self.lineEdit_2)
        self.horizontalLayout_2.addWidget(self.frame2)
        self.runTest = QtWidgets.QPushButton(self.waveParameters)
        self.runTest.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"QPushButton {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #f6f7fa, stop: 1 #dadbde);\n"
"    border: 1px solid #76797C;\n"
"    padding: 5px;\n"
"    border-radius: 4px;\n"
"    color: black;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #dadbde, stop: 1 #f6f7fa);\n"
"}")
        self.runTest.setObjectName("runTest")
        self.horizontalLayout_2.addWidget(self.runTest)
        self.verticalLayout_9.addWidget(self.waveParameters)
        self.frame3 = QtWidgets.QFrame(self.input)
        self.frame3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame3.setObjectName("frame3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.frame3)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.frame_3)
        self.label_4.setStyleSheet("font: 75 14pt \"Times New Roman\";\n"
"")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.frame_3)
        self.label_5.setStyleSheet("font: 75 14pt \"Times New Roman\";\n"
"")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.verticalLayout.addWidget(self.frame_3)
        self.frame_2 = QtWidgets.QFrame(self.frame3)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout_3.addWidget(self.lineEdit_5)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_3.addWidget(self.lineEdit_4)
        self.verticalLayout.addWidget(self.frame_2)
        self.pushButton = QtWidgets.QPushButton(self.frame3)
        self.pushButton.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"QPushButton {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #f6f7fa, stop: 1 #dadbde);\n"
"    border: 1px solid #76797C;\n"
"    padding: 5px;\n"
"    border-radius: 4px;\n"
"    color: black;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.verticalLayout_9.addWidget(self.frame3)
        self.frame_4 = QtWidgets.QFrame(self.input)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.waveparams = QtWidgets.QFrame(self.frame_4)
        self.waveparams.setStyleSheet("font: 75 12pt \"Times New Roman\";\n"
"\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(36, 206, 255, 255), stop:1 rgba(255, 255, 255, 255));")
        self.waveparams.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.waveparams.setFrameShadow(QtWidgets.QFrame.Raised)
        self.waveparams.setObjectName("waveparams")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.waveparams)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_9 = QtWidgets.QLabel(self.waveparams)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout.addWidget(self.label_9)
        self.label_8 = QtWidgets.QLabel(self.waveparams)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        self.label_6 = QtWidgets.QLabel(self.waveparams)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout.addWidget(self.label_6)
        self.label_7 = QtWidgets.QLabel(self.waveparams)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.verticalLayout_5.addWidget(self.waveparams)
        self.paramsInput = QtWidgets.QFrame(self.frame_4)
        self.paramsInput.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.paramsInput.setFrameShadow(QtWidgets.QFrame.Raised)
        self.paramsInput.setObjectName("paramsInput")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.paramsInput)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.paramsInput)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.horizontalLayout_5.addWidget(self.lineEdit_6)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.paramsInput)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.horizontalLayout_5.addWidget(self.lineEdit_9)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.paramsInput)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.horizontalLayout_5.addWidget(self.lineEdit_8)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.paramsInput)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.horizontalLayout_5.addWidget(self.lineEdit_7)
        self.verticalLayout_5.addWidget(self.paramsInput)
        self.verticalLayout_9.addWidget(self.frame_4)
        self.horizontalLayout_11.addWidget(self.input)
        self.Figutes = QtWidgets.QFrame(self.frame)
        self.Figutes.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Figutes.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Figutes.setObjectName("Figutes")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.Figutes)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Fig1 = QtWidgets.QFrame(self.Figutes)
        self.Fig1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Fig1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Fig1.setObjectName("Fig1")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.Fig1)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_10 = QtWidgets.QLabel(self.Fig1)
        self.label_10.setStyleSheet("font: 75 10pt \"Times New Roman\";\n"
"")
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_7.addWidget(self.label_10)
        self.frame_7 = QtWidgets.QFrame(self.Fig1)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_7.addWidget(self.frame_7)
        self.verticalLayout_2.addWidget(self.Fig1)
        self.frame_6 = QtWidgets.QFrame(self.Figutes)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_11 = QtWidgets.QLabel(self.frame_6)
        self.label_11.setStyleSheet("font: 75 10pt \"Times New Roman\";\n"
"")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_8.addWidget(self.label_11)
        self.frame_5 = QtWidgets.QFrame(self.frame_6)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_8.addWidget(self.frame_5)
        self.verticalLayout_2.addWidget(self.frame_6)
        self.cameracatch = QtWidgets.QFrame(self.Figutes)
        self.cameracatch.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cameracatch.setFrameShadow(QtWidgets.QFrame.Raised)
        self.cameracatch.setObjectName("cameracatch")
        self.cameracatch.setStyleSheet("background-color: gray;")
       
        self.pushButton_3 = QtWidgets.QPushButton(self.Figutes)
        self.pushButton_3.setStyleSheet("font: 75 14pt \"Times New Roman\";\n"
"QPushButton {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #f6f7fa, stop: 1 #dadbde);\n"
"    border: 1px solid #76797C;\n"
"    padding: 5px;\n"
"    border-radius: 4px;\n"
"    color: black;\n"
"}")
        self.pushButton_3.setObjectName("pushButton_3")

        self.verticalLayout_2.setStretch(0, 3)
        self.verticalLayout_2.setStretch(1, 3)
        self.verticalLayout_2.setStretch(2, 3)
        self.horizontalLayout_11.addWidget(self.Figutes)
        self.recognition = QtWidgets.QFrame(self.frame)
        self.recognition.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.recognition.setFrameShadow(QtWidgets.QFrame.Raised)
        self.recognition.setObjectName("recognition")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.recognition)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_12 = QtWidgets.QLabel(self.recognition)
        self.label_12.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(36, 206, 255, 255), stop:1 rgba(255, 255, 255, 255));")
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_6.addWidget(self.label_12)
        self.filePath = QtWidgets.QFrame(self.recognition)
        self.filePath.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.filePath.setFrameShadow(QtWidgets.QFrame.Raised)
        self.filePath.setObjectName("filePath")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.filePath)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_13 = QtWidgets.QLabel(self.filePath)
        self.label_13.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(36, 206, 255, 255), stop:1 rgba(255, 255, 255, 255));")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_7.addWidget(self.label_13)
        self.pushButton_2 = QtWidgets.QPushButton(self.filePath)
        self.pushButton_2.setStyleSheet("font: 75 14pt \"Times New Roman\";\n"
"QPushButton {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #f6f7fa, stop: 1 #dadbde);\n"
"    border: 1px solid #76797C;\n"
"    padding: 5px;\n"
"    border-radius: 4px;\n"
"    color: black;\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_7.addWidget(self.pushButton_2)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.filePath)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.verticalLayout_7.addWidget(self.lineEdit_10)
        self.verticalLayout_6.addWidget(self.filePath)
        self.frame_8 = QtWidgets.QFrame(self.recognition)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_8.setObjectName("verticalLayout_8")

        self.verticalLayout_8.addWidget(self.pushButton_3)
        self.label_14 = QtWidgets.QLabel(self.frame_8)
        self.label_14.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(36, 206, 255, 255), stop:1 rgba(255, 255, 255, 255));")
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_8.addWidget(self.label_14)
        self.comboBox = QtWidgets.QComboBox(self.frame_8)
        self.comboBox.setStyleSheet("font: 75 12pt \"Times New Roman\";\n"
"QPushButton {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #f6f7fa, stop: 1 #dadbde);\n"
"    border: 1px solid #76797C;\n"
"    padding: 5px;\n"
"    border-radius: 4px;\n"
"    color: black;\n"
"}")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout_8.addWidget(self.comboBox)
        self.pushButton_4 = QtWidgets.QPushButton(self.filePath)
        self.pushButton_4.setStyleSheet("font: 75 14pt \"Times New Roman\";\n"
"QPushButton {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #f6f7fa, stop: 1 #dadbde);\n"
"    border: 1px solid #76797C;\n"
"    padding: 5px;\n"
"    border-radius: 4px;\n"
"    color: black;\n"
"}")
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_8.addWidget(self.pushButton_4)
        self.verticalLayout_6.addWidget(self.frame_8)

        self.frame_9 = QtWidgets.QFrame(self.recognition)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.frame_9)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_15 = QtWidgets.QLabel(self.frame_9)
        self.label_15.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 75 16pt \"Times New Roman\";\n"
"")
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_10.addWidget(self.label_15)
        self.label_16 = QtWidgets.QLabel(self.frame_9)
        self.label_16.setStyleSheet("font: 75 16pt \"Times New Roman\";\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 189, 167, 250), stop:1 rgba(255, 255, 255, 255));")
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_10.addWidget(self.label_16)
        self.verticalLayout_6.addWidget(self.frame_9)
        self.verticalLayout_6.addWidget(self.cameracatch)
        self.horizontalLayout_11.addWidget(self.recognition)
        self.horizontalLayout_11.setStretch(0, 4)
        self.horizontalLayout_11.setStretch(1, 5)
        self.horizontalLayout_11.setStretch(2, 5)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.goodLuck.setText(_translate("Form", "Good luck with your experiment!"))
        self.goodLuck_2.setText(_translate("Form", "Power controller"))
        self.RRAM.setText(_translate("Form", "RRAM"))
        self.DC.setText(_translate("Form", "DC"))
        self.Triangle.setText(_translate("Form", "Triangle"))
        self.label.setText(_translate("Form", "Amplitude"))
        self.label_2.setText(_translate("Form", "Points"))
        self.label_3.setText(_translate("Form", "Cycles"))
        self.runTest.setText(_translate("Form", "Run"))
        self.label_4.setText(_translate("Form", "Name for J-V"))
        self.label_5.setText(_translate("Form", "Name for log/nJ-V"))
        self.pushButton.setText(_translate("Form", "Save"))
        self.label_9.setText(_translate("Form", "Vread(Unused)"))
        self.label_8.setText(_translate("Form", "highhold"))
        self.label_6.setText(_translate("Form", "lowhold"))
        self.label_7.setText(_translate("Form", "readhold"))
        self.label_10.setText(_translate("Form", "J-V"))
        self.label_11.setText(_translate("Form", "logJ-V"))
        self.pushButton_3.setText(_translate("Form", "Save figure"))
        self.label_12.setText(_translate("Form", "Edge recognition"))
        self.label_13.setText(_translate("Form", "Target file"))
        self.pushButton_2.setText(_translate("Form", "Browse"))
        self.pushButton_4.setText(_translate("Form", "Calculate"))
        self.label_14.setText(_translate("Form", "Recognition mode"))
        self.comboBox.setItemText(0, _translate("Form", "Automaticlly"))
        self.comboBox.setItemText(1, _translate("Form", "Line"))
        self.comboBox.setItemText(2, _translate("Form", "Manually"))
        self.label_15.setText(_translate("Form", "Diameter is"))
        
import res_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
