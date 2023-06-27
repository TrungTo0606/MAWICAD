import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 
import os
from GUI.GUI_AI_v3 import *
import numpy as np
from tensorflow import keras
from lib.camera import*
# from queue import Queue
from imgProcess.imgProcessing_R3_v5 import*


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.welds_count = 0
        self.frame = None
        self.cutFrame = False
        self.weld = []
        self.score_list = []
        self.webcam.clicked.connect(self.loadcamera)
        self.capture.clicked.connect(self.cutFace)
        self.save.clicked.connect(self.save_image)
        self.reset.clicked.connect(self.getreset)
        self.toolButton.clicked.connect(self.selectDirectory)
        self.Score.clicked.connect(self.show_score)  
        self.spinBox.valueChanged.connect(self.onSpinBoxValueChanged)    

    def onSpinBoxValueChanged(self, value_weld):
        if value_weld != 0 and value_weld < 7:
            self.welds_count = value_weld
        else:
            self.notify.setText("Please set number of plate ")

    def loadcamera(self):
        cameraCnt, cameraList = enumCameras()
        if cameraCnt is None:
            self.notify.setText("Discovery no camera")
            return -1
        
        # print camera info
        for index in range(0, cameraCnt):
            camera = cameraList[index]
            print("\nCamera Id = " + str(index))
            print("Key           = " + str(camera.getKey(camera)))
            print("vendor name   = " + str(camera.getVendorName(camera)))
            print("Model  name   = " + str(camera.getModelName(camera)))
            print("Serial number = " + str(camera.getSerialNumber(camera)))
        
        camera = cameraList[0]

        self.streamSource = CameraStatus(camera, status=True)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(50)

        self.notify.setText("Camera conected, please press CAPTURE")

    def update_camera(self):
        self.frame = getFrame(self.streamSource)
        
        self.update_frame()

    def update_frame(self):
        self.setPhoto(self.frame)

    def setPhoto(self, image):
        frame_size = self.Framemain.size()
        # print(image.dtype)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (frame_size.width(), frame_size.height()))
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_RGB888)
        self.Framemain.setPixmap(QtGui.QPixmap.fromImage(img))

    def cut(self, frame, cuttedFace):
        if cuttedFace is None:
            return
        image_cut = cv2.cvtColor(cuttedFace,cv2.COLOR_BGR2RGB) #CuttedFace là hình cắt từ frame
        frame_size = frame.size()
        image_cut = cv2.resize(image_cut,(frame_size.width(), frame_size.height()))
        img_cut = QtGui.QImage(image_cut,image_cut.shape[1],image_cut.shape[0],image_cut.strides[0],QtGui.QImage.Format_RGB888)
        frame.setPixmap(QtGui.QPixmap.fromImage(img_cut))
        
    def cutFace(self):
        if self.frame is None:
            return
        if self.welds_count == 0:
            self.notify.setText("Please set number of plate ")
            return
        self.frame, self.weld = imgProcessing(self.frame
                                              ,self.welds_count)
        self.update_frame()

        err = 0
        for weld in self.weld:
            if weld is None:
                err +=1
                if err == 5:
                    self.notify.setText("Please check the number of plate ")
                    return
        
        if self.weld[0] is None:
            self.Frame1.clear()
        else:
            self.cut(self.Frame1,self.weld[0])
        if self.weld[1] is None:
            self.Frame2.clear()
        else:    
            self.cut(self.Frame2,self.weld[1])
        if self.weld[2] is None:
            self.Frame3.clear()
        else:
            self.cut(self.Frame3,self.weld[2])
        if self.weld[3] is None:
            self.Frame4.clear()
        else:
            self.cut(self.Frame4,self.weld[3])
        if self.weld[4] is None:
            self.Frame5.clear()
        else:
            self.cut(self.Frame5,self.weld[4])
        
        self.notify.setText("Press ASSESS for receive feedback")
        
    def show_score(self):

        self.notify.setText("Please waiting for feedback")
        self.score_list = []
        
        score_1 = self.check_score(self.weld[0])
        self.score1.setText(score_1)
        self.score_list.append(score_1)
        
        score_2 = self.check_score(self.weld[1])
        self.score2.setText(score_2)
        self.score_list.append(score_2)
        
        score_3 = self.check_score(self.weld[2])
        self.score3.setText(score_3)
        self.score_list.append(score_3)
        
        score_4 = self.check_score(self.weld[3])
        self.score4.setText(score_4)
        self.score_list.append(score_4)
        
        score_5 = self.check_score(self.weld[4])
        self.score5.setText(score_5)
        self.score_list.append(score_5) 
    
        self.notify.setText("Assess completed")
    
            
    def get_index(self):
        # Thiết lập đường dẫn đến thư mục lưu
        self.save_dir = self.save_address.text()
        # Lấy danh sách các tệp trong thư mục lưu và sắp xếp theo thứ tự giảm dần theo thời gian sửa đổi
        files = sorted(os.listdir(self.save_dir), key=lambda x: int(x.split('_')[0]), reverse=True)

        # Lấy tên file cuối cùng trong danh sách và hiển thị trên QLabel
        if len(files) > 0:
            self.index = files[0]
            self.index = int(self.index.split("_")[0])
            #print(self.index)
        else:
            self.index = 0
        
    def selectDirectory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory Save")
        if not directory: # Nếu directory không có giá trị thì return
            return
        self.save_address.setText(directory)
        self.get_index()
    
    
    def check_score(self,image):
        if image is None:
            return
        image = cv2.resize(image, (96, 96))
        image = np.expand_dims(image, axis=0)

        # # Nhúng tệp h5 thành tệp dữ liệu nhúng
        # with importlib_resources.path("WeldClassication_4classes_v2.h5") as h5_path:
        #     # Sử dụng đường dẫn tệp dữ liệu nhúng
        #     model = keras.models.load_model(str(h5_path))
        
        model = keras.models.load_model("WeldClassication_4classes_v2.h5")
        # Dự đoán nhãn của ảnh test
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])

        # In kết quả
        class_names = ['A', 'B', 'C', 'D']
        return class_names[predicted_class]
        
                
    
    def save_frame(self,save_path,image,index,score):
        image = cv2.resize(image,(600,160))
        image_path = "{}/{}_{}.jpg".format(save_path,index,score)
        cv2.imwrite(image_path, image)
    
    def save_image(self):
        if self.save_address.text() == "Select folder save data":
            self.selectDirectory()
        if self.save_address.text() != "Select folder save data":
            save_path = self.save_address.text()
            self.get_index()
            #Save frame1
            if self.Frame1.pixmap() != None:
                    self.index += 1
                    self.save_frame(save_path,self.weld[0],self.index,self.score_list[0])
            #Save frame2
            if self.Frame2.pixmap() != None:
                    self.index += 1
                    self.save_frame(save_path,self.weld[1],self.index,self.score_list[1])
            #Save frame3
            if self.Frame3.pixmap() != None:
                    self.index += 1
                    self.save_frame(save_path,self.weld[2],self.index,self.score_list[2])
            #Save frame4
            if self.Frame4.pixmap() != None:
                    self.index += 1
                    self.save_frame(save_path,self.weld[3],self.index,self.score_list[3])
            #Save frame5
            if self.Frame5.pixmap() != None:
                    self.index += 1
                    self.save_frame(save_path,self.weld[4],self.index,self.score_list[4])

        self.notify.setText("Save completed")
    
    def rev_frame(self,groupbox,frame):
        for radiobutton in groupbox.findChildren(QtWidgets.QRadioButton):
            if radiobutton.isChecked():
                radiobutton.setAutoExclusive(False)
                radiobutton.setChecked(False)
                break
        frame.clear()
    
    def getreset(self):
        self.score1.setText("")
        self.score2.setText("")
        self.score3.setText("")
        self.score4.setText("")
        self.score5.setText("")
        self.Frame1.clear()
        self.Frame2.clear()
        self.Frame3.clear()
        self.Frame4.clear()
        self.Frame5.clear()
        self.weld = []
        while len(self.weld) < 5:
            self.weld.append(None)
        self.notify.setText("Reset completed") 

    def closeEvent(self, event):
        event.accept()
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())    