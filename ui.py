import tobii_research as tr
import time
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from matplotlib import pyplot as plt
from threading import *
import SimpleITK as sitk

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5 import uic, QtWidgets


def truncate_hu(image_array):
    image_array[image_array > 250] = 250
    image_array[image_array <-250] = -250
    
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array - min)/(max - min)
    image_array = (image_array*255).astype(np.uint8)
    return image_array 


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = [] # list of QPoint objects
        self.point = None
        self.show_all_points = False
    
    def set_point(self, point):
        self.point = point
        self.update() # redraw the widget
    
    def add_point(self, point):
        self.points.append(point)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self.point is not None:
            painter.drawEllipse(self.point, 15, 15)

        painter.setPen(QPen(Qt.yellow, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self.show_all_points:
            for point in self.points:
                painter.drawPoint(point)

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        # self.setGeometry(0, 0, pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)
        self.update()

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, tracker, save_path):
        super().__init__()
        
        # Load the UI file
        uic.loadUi('ui.ui', self)
        self.layoutwin = self.findChild(QHBoxLayout, 'horizontalLayout')
        self.chooseimg_button = self.findChild(QPushButton, 'Chooseimg_button')
        self.Start_button = self.findChild(QPushButton, 'Start_button')
        self.Stop_button = self.findChild(QPushButton, 'Stop_button')
        self.Showtrack_button = self.findChild(QPushButton, 'Showtrack_button')
        self.cleartrack_button = self.findChild(QPushButton, 'clear_button')
        self.loadSAM_button = self.findChild(QPushButton, 'LoadSAM_button')
        self.onepoint_button = self.findChild(QPushButton, 'onepoint_button')
        self.savemask_button = self.findChild(QPushButton, 'savemask_button')
        # self.Savemask_button = self.findChild(QPushButton, 'Savemask_button')
        self.scrollbar = self.findChild(QtWidgets.QScrollBar, 'ScrollBar')
        self.old_label = self.findChild(QLabel, 'label')
        self.label = ImageLabel(self)
        self.label.setObjectName("label")
        self.layoutwin.replaceWidget(self.old_label, self.label)
        self.old_label.deleteLater()

        # define function of the widgets
        self.chooseimg_button.clicked.connect(self.choose_image)
        self.loadSAM_button.clicked.connect(self.loadsam)
        self.Start_button.clicked.connect(self.start_tracker)
        self.Stop_button.clicked.connect(self.stop_tracker)
        self.cleartrack_button.clicked.connect(self.clear_points)
        self.Showtrack_button.clicked.connect(self.show_track)
        self.savemask_button.clicked.connect(self.savemask)
        self.onepoint_button.clicked.connect(self.kthread)
        self.scrollbar.valueChanged.connect(self.on_scrollbar_value_changed)
        # self.Savemask_button.clicked.connect(self.savemask)

        # Initizlization of variables
        self.tracker = tracker
        self.is_tracking = False
        self.predictor = None
        self.imgpath = None
        self.pixmap = None
        self.savepath = save_path
        self.file = None
        self.maskimg = None
        self.image = None
        self.count = 1
        self.t1 = None
        self.point = None
        self.lock = Lock()
        self.label_top_left_pos = None
        self.image_top_left_pos = None
        self.scale_ratio = None
        self.threadopen = False
        self.med = False
        self.img_set = True
        self.loaded = False
        self.img3d = None
        self.left = []
        self.right = []
        self.imgpos = []
        self.screensize = QApplication.desktop().screenGeometry()

        self.timer = QTimer()
        self.timer.timeout.connect(self.keep_run_sam)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z:
            self.start_tracker()
        elif event.key() == Qt.Key_X:
            self.stop_tracker()
        elif event.key() == Qt.Key_A:
            self.clear_points()
        elif event.key() == Qt.Key_C:
            self.savemask()
        elif event.key() == Qt.Key_Q:
            self.setsliceimg()

    def choose_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'Image files (*.jpg *.png *.jpeg *.nii.gz)')
        self.imgpath = filename
        self.file = filename.split("/")[-1].split(".")[0]
        if filename:
            if filename.split(".")[-1] == 'gz':
                self.med = True
                sitkimage = sitk.ReadImage(filename)
                img = sitk.GetArrayFromImage(sitkimage)
                truncate_hu(img)
                img = normalazation(img)
                self.img3d = img
                self.scrollbar.setMaximum(img.shape[0])
                pixmap = self.show3dimg()

                # get self.label screen position
                label_top_left = self.label.mapToGlobal(QPoint(0,0))
                self.label_top_left_pos = [label_top_left.x(), label_top_left.y()]

                # get image screen position
                self.image_top_left_pos = [self.label_top_left_pos[0]+(self.label.width()-pixmap.width())/2,
                                self.label_top_left_pos[1]+(self.label.height()-pixmap.height())/2]

            else:
                ori_pixmap = QPixmap(filename)
                pixmap = ori_pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
                self.scale_ratio = [ori_pixmap.width()/pixmap.width(), ori_pixmap.height()/pixmap.height()]
                self.pixmap = pixmap
                self.label.setAlignment(Qt.AlignCenter)
                self.label.setPixmap(pixmap)

                # get self.label screen position
                label_top_left = self.label.mapToGlobal(QPoint(0,0))
                self.label_top_left_pos = [label_top_left.x(), label_top_left.y()]

                # get image screen position
                self.image_top_left_pos = [self.label_top_left_pos[0]+(self.label.width()-pixmap.width())/2,
                                self.label_top_left_pos[1]+(self.label.height()-pixmap.height())/2]
    
    def show3dimg(self):
        medimg = self.img3d[self.scrollbar.value(),...]
        medimg = (np.dstack([medimg, medimg, medimg]))
        self.image = medimg
        if self.loaded:
            self.predictor.set_image(self.image)
            self.img_set = True
        ori_pixmap = QPixmap.fromImage((QImage(medimg.data, medimg.shape[1], medimg.shape[0], QImage.Format_RGB888)))
        pixmap = ori_pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        self.scale_ratio = [ori_pixmap.width()/pixmap.width(), ori_pixmap.height()/pixmap.height()]
        self.pixmap = pixmap
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setPixmap(pixmap)
        return pixmap


    def on_scrollbar_value_changed(self, value):
        self.img_set = False
        print('Scrollbar value changed:', value)
        _ = self.show3dimg()



    def setsliceimg(self):
        if self.loaded:
            self.predictor.set_image(self.image)
            self.kthread()

    def kthread(self):
        # self.t1=Thread(target=self.keep_run_sam)
        # self.t1.start()
        # self.threadopen = True
        self.timer.start(10)


    def loadsam(self):
        print("load SAM model")
        self.loaded = True
        sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        image = cv2.imread(self.imgpath)
        if self.med == False:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(self.image)
            self.img_set = True
        else:
            self.on_scrollbar_value_changed(self.scrollbar.value())

    def update_point(self, point, qpoint):
        self.label.set_point(qpoint)
        self.label.add_point(qpoint)
        self.point = point

    def clear_points(self):
        print("clear gaze")
        self.label.point = None
        self.label.set_pixmap(self.pixmap)
        self.label.points.clear()
        self.label.show_all_points = False

    def start_tracker(self):
        print("start recording")
        self.is_tracking = True
        self.tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.run_tracker, as_dictionary=True)
    
    def stop_tracker(self):
        print("stop recording")
        self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.run_tracker)
        self.label.point = None
        time.sleep(1)
        self.label.set_pixmap(self.pixmap)

    def show_track(self):
        print("show track")
        self.label.show_all_points = True
        self.label.update()

    def keep_run_sam(self):
        # while True:
        if self.label.point is not None and self.img_set == True:
            input_point = np.array([self.point])
            input_label = np.array([1])

            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            re_mask=masks.transpose((1,2,0))
            mask3d=(np.dstack([re_mask*0, re_mask*255, re_mask*0]))
            mask3d=mask3d.astype(np.uint8)
            add_image= cv2.addWeighted(self.image, 0.7, mask3d, 0.3, 0.0)
            self.maskimg = mask3d


            qimage = QPixmap.fromImage((QImage(add_image.data, add_image.shape[1], add_image.shape[0], QImage.Format_RGB888)))
            # if not self.med :
            qimage = qimage.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setPixmap(qimage)
        elif self.label.point is None:
            print("thread close")
            self.threadopen = False
                # break

    def savemask(self):
        cv2.imwrite(self.savepath + 'mask{}.jpg'.format(self.count), cv2.cvtColor(self.maskimg, cv2.COLOR_RGB2BGR))
        self.count +=1

    def run_sam(self):
        if self.point is not None:
            input_point = np.array([self.point])
            input_label = np.array([1])

            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            re_mask=masks.transpose((1,2,0))
            mask3d=(np.dstack([re_mask*0, re_mask*255, re_mask*0]))
            mask3d=mask3d.astype(np.uint8)
            add_image= cv2.addWeighted(self.image, 0.7, mask3d, 0.3, 0.0)


            qimage = QImage(add_image.data, add_image.shape[1], add_image.shape[0], QImage.Format_RGB888)
            q_image = QPixmap.fromImage(qimage)
            self.label.set_pixmap(q_image)


    def run_tracker(self, gaze_data):
        self.left.append(list(gaze_data['left_gaze_point_on_display_area']))
        self.right.append(list(gaze_data['right_gaze_point_on_display_area']))
        
        # point is from 0-1, meaning relative position to screen
        point = list(np.mean([list(gaze_data['left_gaze_point_on_display_area']), list(gaze_data['right_gaze_point_on_display_area'])], axis=0))
        # screen_coord is absolute position of point on screen
        screen_coord = [point[0]*(self.screensize.width()), point[1]*(self.screensize.height())]
        # label_coord is absolute position of point on self.label(QLabel)
        label_coord = [screen_coord[0] - self.label_top_left_pos[0],
                       screen_coord[1] - self.label_top_left_pos[1]]
        # scaled-image_coord 
        scalImage_coord = [screen_coord[0] - self.image_top_left_pos[0],
                       screen_coord[1] - self.image_top_left_pos[1]]
        # original image coord
        image_coord = [scalImage_coord[0] * self.scale_ratio[0],
                       scalImage_coord[1] * self.scale_ratio[1]]

        window.update_point([int(image_coord[0]), int(image_coord[1])], 
                            QPoint(int(label_coord[0]), int(label_coord[1])))




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    found_eyetrackers = tr.find_all_eyetrackers()
    my_eyetracker = found_eyetrackers[0]
    save_path = "./save/"
    window = MyWindow(my_eyetracker, save_path)
    window.show()
    sys.exit(app.exec_())
