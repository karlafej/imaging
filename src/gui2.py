import sys
import re
import PyQt5.QtWidgets as qt
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap

class PredictDXA(qt.QWidget):
    '''
    GUI to predict.py
    '''
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.in_folder = '.'
        self.leg_sw = 0
        self.dxa_sw = 0

    def init_ui(self):
        '''
        GUI elements
        '''
        self.setGeometry(300, 300, 1200, 300)
        self.setWindowTitle("Predict DXA")

        self.lbl1 = qt.QLabel('Select input directory', self)
        self.lbl1.setMinimumWidth(220)
        self.input_txt = qt.QLineEdit(self)

        self.btn1 = qt.QPushButton('Browse', self)
        self.btn1.setToolTip('Select input directory')
        self.btn1.clicked.connect(self.choose_dir)

        self.lbl2 = qt.QLabel('Mouse legs are...', self)
        self.s1 = qt.QRadioButton("Not stretched")
        self.s1.setChecked(True)
        self.s1.clicked.connect(lambda: self.btnstate(self.s1))
        self.s2 = qt.QRadioButton("Stretched")
        self.s2.clicked.connect(lambda: self.btnstate(self.s2))
        self.btn_submit = qt.QPushButton('Submit', self)
        self.btn_submit.clicked.connect(lambda: self.call_program(self.in_folder,
                                                                  self.leg_sw,
                                                                  self.dxa_sw))
        self.btn_cancel = qt.QPushButton('Cancel', self)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.end)

        self.lbl3 = qt.QLabel('Folder', self)
        self.box = qt.QCheckBox('Rec')
        self.box.setChecked(False)
        self.box.stateChanged.connect(lambda: self.boxstate(self.box))

        self.output = qt.QTextEdit()

        self.img_label = qt.QLabel()
        self.img = qt.QLabel()
        self.img.setFixedHeight(300)

        # QProcess object for external app
        self.process = QtCore.QProcess(self)
        # QProcess emits `readyRead` when there is data to be read
        self.process.readyRead.connect(lambda: [self.data_ready(), self.set_img()])

        self.process.started.connect(lambda: [self.btn_submit.setEnabled(False),
                                              self.btn_cancel.setEnabled(True)])
        self.process.finished.connect(lambda: [self.btn_submit.setEnabled(True),
                                               self.btn_cancel.setEnabled(False)])
        self.create_grid_layout()
        window_layout = qt.QVBoxLayout()
        window_layout.addWidget(self.horizontal_groupbox)
        self.setLayout(window_layout)

    def create_grid_layout(self):
        '''
        arrange gui elements
        '''
        self.horizontal_groupbox = qt.QGroupBox()
        grid = qt.QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self.lbl1, 1, 0)
        grid.addWidget(self.input_txt, 2, 0, 1, 3)
        grid.addWidget(self.btn1, 2, 3)
        grid.addWidget(self.lbl2, 4, 0, 1, 2)
        grid.addWidget(self.s1, 5, 0, 1, 2)
        grid.addWidget(self.s2, 6, 0, 1, 2)
        grid.addWidget(self.lbl3, 4, 2, 1, 2)
        grid.addWidget(self.box, 5, 2, 1, 2)
        grid.addWidget(self.btn_submit, 8, 0)
        grid.addWidget(self.btn_cancel, 8, 3)
        grid.addWidget(self.output, 1, 4, 10, 10)
        grid.addWidget(self.img, 12, 12, 2, 2)
        grid.addWidget(self.img_label, 14, 12)

        self.horizontal_groupbox.setLayout(grid)

    def choose_dir(self):
        '''
        Select input directory
        '''
        current = str(self.input_txt.text())

        if current:
            input_dir = qt.QFileDialog.getExistingDirectory(None, 'Select a folder:', current)
        else:
            input_dir = qt.QFileDialog.getExistingDirectory(None, 'Select a folder:')
        self.input_txt.setText(input_dir)
        self.in_folder = input_dir

    def btnstate(self, b):
        '''
        Read state of Legs stretched/unstretched radiobuttons
        '''
        if b.text() == "Not stretched":
            if b.isChecked:
                self.leg_sw = 0
        if b.text() == "Stretched":
            if b.isChecked:
                self.leg_sw = 1

    def boxstate(self, b):
        '''
        dxa checkbox state
        '''
        if b.isChecked():
            self.dxa_sw = 1
        else:
            self.dxa_sw = 0

    def data_ready(self):
        '''
        read stdout from running process, put it into textbox
        '''
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(str(self.process.readAll(), 'utf-8'))
        self.output.ensureCursorVisible()

    def call_program(self, in_folder, leg, dxa):
        '''
        run the process
        `start` takes the exec and a list of arguments
        '''
        if leg:
            stretch = '-s'
        else:
            stretch = ''
        if dxa:
            fld = '-d'
        else:
            fld = ''
        in_folder = '"' + in_folder + '"'
        cmd = 'python predict.py -i ' + in_folder + stretch + fld
        print(in_folder)
        #self.process.start('pwd')
        self.process.start(cmd)
        #self.process.start('python ./predict.py', ['-i', in_folder, stretch, fld])
        #self.process.start('python test2.py')

    def end(self):
        '''
        terminate running process
        '''
        self.process.terminate()

    def set_img(self):
        '''
        Read image path from stdout
        Show the image
        '''
        txt = self.output.toPlainText()
        lst = find_all(txt)
        if lst:
            i_start = lst[-1]+13
            i_end = txt.find(".bmp", i_start)+4
            img_path = txt[i_start:i_end]
            self.img.resize(300, 300)
            pixmap = QPixmap(img_path)
            self.img_label.setText("First image: " + img_path.split("/")[-1])
            self.img.setPixmap(pixmap.scaled(self.img.size(), QtCore.Qt.KeepAspectRatio))


def find_all(txt, substring="First image:"):
    '''
    Input: string, substring
    Output: list of starting indices of all non-overlapping
            occurences of the substring in string
    '''
    lst = [s.start() for s in re.finditer(substring, txt)]
    return lst


if __name__ == '__main__':

    app = qt.QApplication(sys.argv)
    ex = PredictDXA()
    ex.show()
    sys.exit(app.exec_())
