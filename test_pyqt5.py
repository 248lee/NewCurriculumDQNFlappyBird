import sys
from PyQt5.QtWidgets import QApplication, QWidget, QRadioButton, QVBoxLayout, QHBoxLayout, QButtonGroup

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Radio Button Example'
        self.left = 100
        self.top = 100
        self.width = 320
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()

        # First set of radio buttons
        self.radio1a = QRadioButton('Option 1a')
        self.radio1b = QRadioButton('Option 1b')
        self.radio1c = QRadioButton('Option 1c')
        hbox1.addWidget(self.radio1a)
        hbox1.addWidget(self.radio1b)
        hbox1.addWidget(self.radio1c)

        # Second set of radio buttons
        self.radio2a = QRadioButton('Option 2a')
        self.radio2b = QRadioButton('Option 2b')
        self.radio2c = QRadioButton('Option 2c')
        hbox2.addWidget(self.radio2a)
        hbox2.addWidget(self.radio2b)
        hbox2.addWidget(self.radio2c)

        # Create button groups for each set of radio buttons
        group1 = QButtonGroup(self)
        group1.addButton(self.radio1a)
        group1.addButton(self.radio1b)
        group1.addButton(self.radio1c)

        group2 = QButtonGroup(self)
        group2.addButton(self.radio2a)
        group2.addButton(self.radio2b)
        group2.addButton(self.radio2c)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        self.setLayout(vbox)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
