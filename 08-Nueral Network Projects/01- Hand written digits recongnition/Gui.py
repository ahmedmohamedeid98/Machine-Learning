# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form3.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1236, 648)
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setGeometry(QtCore.QRect(870, 420, 131, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(1010, 400, 191, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setEnabled(False)
        self.pushButton.setGeometry(QtCore.QRect(660, 420, 141, 40))
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(360, 10, 601, 51))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.line = QtWidgets.QFrame(Form)
        self.line.setGeometry(QtCore.QRect(840, 120, 21, 341))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(620, 180, 211, 191))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setEnabled(True)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 110, 171, 40))
        self.pushButton_3.setObjectName("pushButton_3")
        self.accuarcy_label = QtWidgets.QLabel(Form)
        self.accuarcy_label.setGeometry(QtCore.QRect(50, 300, 161, 271))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.accuarcy_label.setFont(font)
        self.accuarcy_label.setFrameShape(QtWidgets.QFrame.Box)
        self.accuarcy_label.setText("")
        self.accuarcy_label.setScaledContents(False)
        self.accuarcy_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.accuarcy_label.setObjectName("accuarcy_label")
        self.ploting_label = QtWidgets.QLabel(Form)
        self.ploting_label.setGeometry(QtCore.QRect(230, 300, 361, 271))
        self.ploting_label.setFrameShape(QtWidgets.QFrame.Box)
        self.ploting_label.setText("")
        self.ploting_label.setObjectName("ploting_label")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(50, 270, 121, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(230, 270, 151, 28))
        self.label_7.setObjectName("label_7")
        self.progressBar = QtWidgets.QProgressBar(Form)
        self.progressBar.setGeometry(QtCore.QRect(50, 210, 261, 30))
        self.progressBar.setAutoFillBackground(True)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(50, 180, 111, 28))
        self.label_4.setObjectName("label_4")
        self.show_info = QtWidgets.QPushButton(Form)
        self.show_info.setGeometry(QtCore.QRect(660, 550, 141, 40))
        self.show_info.setObjectName("show_info")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(830, 500, 371, 131))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setText("")
        self.label_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_5.setObjectName("label_5")
        self.Digites_plot = QtWidgets.QLabel(Form)
        self.Digites_plot.setGeometry(QtCore.QRect(872, 120, 331, 251))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setItalic(True)
        self.Digites_plot.setFont(font)
        self.Digites_plot.setFrameShape(QtWidgets.QFrame.Box)
        self.Digites_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.Digites_plot.setObjectName("Digites_plot")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_2.setText(_translate("Form", "Recognize"))
        self.label.setText(_translate("Form", "Network output: "))
        self.pushButton.setText(_translate("Form", "Brows"))
        self.label_2.setText(_translate("Form", "Hand Written digit Recognision"))
        self.label_3.setText(_translate("Form", "load image ..."))
        self.pushButton_3.setText(_translate("Form", "Train Network"))
        self.label_6.setText(_translate("Form", "Accuracy"))
        self.label_7.setText(_translate("Form", "Visualize accuracy"))
        self.label_4.setText(_translate("Form", "training Prograss"))
        self.show_info.setText(_translate("Form", "show info"))
        self.Digites_plot.setText(_translate("Form", "Digits Weight plot"))

