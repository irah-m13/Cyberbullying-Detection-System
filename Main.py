# importing libraries
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import sys
global tui
sites_to_block = [
    "www.instagram.com",
    "https://www/instagram.com",
    "instagram.com"
]
Window_host = r"C:\Windows\System32\drivers\etc\hosts"
default_hoster = Window_host
redirect = "127.0.0.1"
with open(default_hoster, "r+") as hostfile:
    hosts = hostfile.readlines()
    hostfile.seek(0)
    for host in hosts:
        if not any(site in host for site in sites_to_block):
            hostfile.write(host)
    hostfile.truncate()
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPalette, QBrush
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget ,QVBoxLayout ,QPushButton ,QTextEdit,QFileDialog
global fname
from csv import writer
import csv
import pickle
global fry
from difflib import SequenceMatcher
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image  
from pnslib import utils
tui=0
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def sif(dc,dv):
    img1= cv2.imread(dc)
    img1= cv2.resize(img1, (250, 250))
    img2 = cv2.imread(dv)
    img2= cv2.resize(img2, (250, 250))

    # img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # convert the images from bgr to rgb
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # print(img1_rgb.shape)

    # show the images for reference
    plt.figure(figsize=(20,10))
    plt.imshow(img1)
    plt.title('Image 1')
    plt.show()

    plt.figure(figsize=(20,10))
    plt.imshow(img2)
    plt.title('Image 2')
    plt.show()
    gray= cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    print(gray.shape)

    plt.figure(figsize=(20,10))
    plt.imshow(gray,cmap='gray', vmin=0, vmax=255)
    plt.show()

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)

    keypoints=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg',img1)

    print(keypoints.shape)
    plt.figure(figsize=(20,10))
    plt.imshow(keypoints)
    plt.title('Keypoints of Image 1 for reference')
    plt.show()

    # for face detection
    face_cascade = cv2.CascadeClassifier(utils.get_haarcascade_path('haarcascade_frontalface_default.xml'))

    # images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2= cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    gray=[gray1,gray2]

    # detect faces in the 2 images
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    roi_gray=[]
    roi_color=[]

    size=gray1.shape

    # crop out only the face of the first and second images
    for (x,y,w,h) in faces1:

        extra=int(w/6)
        x1=max(0,x-extra)
        y1=max(0,y-extra)
        x2=min(size[1],x1+2*extra+w)
        y2=min(size[0],y1+2*extra+w)

        img1 = cv2.rectangle(img1,(x1,y1),(x2-1,y2-1),(0,0,255),4)
        roi_gray .append(gray1[y1:y2, x1:x2])
        roi_color .append(img1[y1:y2, x1:x2])

    if len(faces1)==0:
      roi_gray .append(gray1)
      roi_color .append(img1)
        
    size=gray2.shape
    for (x,y,w,h) in faces2:

        extra=int(w/6)
        x1=max(0,x-extra)
        y1=max(0,y-extra)
        x2=min(size[1],x1+2*extra+w)
        y2=min(size[0],y1+2*extra+w)

        img2 = cv2.rectangle(img2,(x1,y1),(x2-1,y2-1),(0,0,255),4)
        roi_gray .append(gray2[y1:y2, x1:x2])
        roi_color .append(img2[y1:y2, x1:x2])

    if len(faces2)==0:
      roi_gray .append(gray2)
      roi_color .append(img2)

    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # roi_color=cv2.cvtColor(roi_color,cv2.COLOR_BGR2RGB)

    # plot the cropped out grayscale images of the originals
    plt.figure(figsize=(20,10))
    plt.imshow(roi_gray[0],cmap='gray', vmin=0, vmax=255)
    plt.title('ROI of image 1')
    plt.show()

    plt.figure(figsize=(20,10))
    plt.imshow(roi_gray[1],cmap='gray', vmin=0, vmax=255)
    plt.title('ROI of image 2')
    plt.show()

    # using SIFT detect the feature descriptors of the 2 images
    import time
    start_time = time.time()
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(roi_gray[0],None)
    kp2, des2 = sift.detectAndCompute(roi_gray[1],None)

    # create a bruteforce matcher
    bf = cv2.BFMatcher(
        # cv2.NORM_L2, 
        # crossCheck=True
        )

    # Match descriptors.
    # matches = bf.match(des1,des2)
    matches=bf.knnMatch(des1,des2,k=2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    len(matches[0])

    # Apply ratio test to filter out only the good matches
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            
    print(len(matches))
    print(len(good))

    # Draw first 10 matches.
    # img3=cv2.drawMatches(roi_gray[0],kp1,roi_gray[0],kp2,matches,None,flags=2)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(roi_gray[0],kp1,roi_gray[1],kp2,good,None,flags=2)

    # decide whether the images are a match or not based on the number of good matches.
    # Yes, crude but a good starting intuition
    ty=0
    if len(good)>=19:
      print("It's a Match")
      ty=1
    else:
      print("Not a Match")
      ty=0
    print("--- %s seconds ---" % (time.time() - start_time))
    
    plt.figure(figsize=(40,20))
    plt.imshow(img3)
    plt.show()
    return ty
# creating a class
# that inherits the QDialog class
class Window(QDialog):

	# constructor
	def __init__(self):
		super(Window, self).__init__()

		# setting window title
		self.setWindowTitle("Python")

		# setting geometry to the window
		self.setGeometry(100, 100, 600, 400)
		
		
		# creating a group box
		self.formGroupBox = QGroupBox("INSTAGRAM")
		self.formGroupBox.setAlignment(Qt.AlignCenter)
		self.formGroupBox.setStyleSheet('QGroupBox:title {color:white;}')

		# creating spin box to select age
		self.ageSpinBar = QSpinBox()

		# creating combo box to select degree
		self.marComboBox = QComboBox()

		# adding items to the combo box
		self.marComboBox.addItems(["MARRIED", "UNMARRIED"])

		# creating a line edit
		self.nameLineEdit = QLineEdit()
		self.aboutEdit = QLineEdit()
		self.addressEdit = QLineEdit()
		self.DobEdit = QLineEdit()
		self.mobileEdit = QLineEdit()
		self.btn1 = QPushButton("UPLOAD PROFILE PICTURE")
		self.btn1.clicked.connect(self.getfile)
		
		# calling the method that create the form
		self.createForm()

		# creating a dialog button for ok and cancel
		self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

		# adding action when form is accepted
		self.buttonBox.accepted.connect(self.getInfo)

		# adding action when form is rejected
		self.buttonBox.rejected.connect(self.reject)
		
		# creating a vertical layout
		mainLayout = QVBoxLayout()
		mainLayout.setContentsMargins(0,30,0,0)
		mainLayout.setSpacing(5)
		# adding form group box to the layout
		mainLayout.addWidget(self.formGroupBox)
		mainLayout.addWidget(self.btn1)
		# adding button box to the layout
		mainLayout.addWidget(self.buttonBox)
		# setting lay out
		self.setLayout(mainLayout)
		# creating label
		self.label = QLabel(self)
		# loading image
		self.pixmap = QPixmap('test2.png')
		# adding image to label
		self.label.setPixmap(self.pixmap)
		self.label.move(285, 0)
		# Optional, resize label to image size
		self.label.resize(30,
                          30)
		# show all the widgets
		self.show()
		
		oImage = QImage("test.jpg")
		sImage = oImage.scaled(QSize(600,400))                   # resize Image to widgets size
		palette = QPalette()
		palette.setBrush(QPalette.Window, QBrush(sImage))
		self.setPalette(palette)
		self.show()
		
	def getfile(self):
		global fname        
		fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Image files (*.jpg *.gif)")
        # get info method called when form is accepted
	def getInfo(self):

		global fname # printing the form information
		a=self.nameLineEdit.text()
		b=self.marComboBox.currentText()
		c=self.ageSpinBar.text()
		d=self.aboutEdit.text()
		e=self.DobEdit.text()
		f=self.mobileEdit.text()
		g=self.addressEdit.text()
		h=str(fname[0])
		db=[a,b,c,d,e,f,g,h]
		with open('database.csv',"r") as f:
		    reader = csv.reader(f,delimiter = ",")
		    data = list(reader)
		    row_count = len(data)
		print(row_count)
		if row_count<=1:
		    with open('database.csv', 'a') as f_object:
		        writer_object = writer(f_object)
		        writer_object.writerow(db)
		        f_object.close()
		        print("ACCOUNT CREATED SUCESSFULLY")
		    # closing the window
		    self.close()
		else:
		    print("ML")
		    print(db)
		    self.close()
		    global fry
		    with open('database.csv') as file_obj:
		        reader_obj = csv.reader(file_obj)
		        for row in reader_obj:
		            try:
		            
		                #print(row[2])
		                pred=[]
		                aa=similar(db[0],row[0])
		                bb=similar(db[1],row[1])
		                cc=similar(db[2],row[2])
		                dd=similar(db[3],row[3])
		                ee=similar(db[4],row[4])
		                ff=similar(db[5],row[5])
		                gg=similar(db[6],row[6])
		                pred=[aa,bb,cc,dd,ee,ff,gg]
		                filename = 'svm.sav'
		                loaded_model = pickle.load(open(filename, 'rb'))
		                person_reports = [[aa,bb,cc,dd,ee,ff,gg]]
		                predicteds = loaded_model.predict(person_reports)
		                #print(int(predicted))
		                import tensorflow as tf
		                model_dir = "./cnn_model"
		                #filename1 = 'cnn.sav'
		                loaded_model1 = tf.keras.models.load_model(model_dir)
		                predicted = loaded_model1.predict(person_reports)
		                prediction = list(predicted[0])
		                print(int(prediction.index(max(prediction))))
		                pred2=int(prediction.index(max(prediction)))
		                #print(pred2)
		                fry=int(predicteds)*pred2
		                #print(fry)
		                fry2=sif(db[7],row[7])
		                
		
		                if fry==1 or fry2==1:
		                    global tui
		                    tui=1
		                    break
		        #file_obj.close()        
		                    
		                    
		                    
		                    
		            except Exception as e:
		                print(e)
		    
		    if tui==0:
		        print("ACCOUNT CREATED SUCESSFULLY")
		        import tkinter.messagebox
		        tkinter.messagebox.showinfo("RESULT","ACCOUNT CREATED SUCESSFULLY")
		        
		        with open('database.csv', 'a') as f_object:
		            writer_object = writer(f_object)
		            writer_object.writerow(db)
		            f_object.close()
		    else:
		        print("FAKE PROFILE")
		        import tkinter.messagebox
		        tkinter.messagebox.showinfo("RESULT","FAKE PROFILE")
		        
		        with open(default_hoster, "r+") as hostfile:
		            hosts = hostfile.read()
		            for site in sites_to_block:
		                if site not in hosts:
		                    hostfile.write(redirect + " " + site + "\n")
		    #print(similar(db,row))
	# creat form method
	def createForm(self):

		# creating a form layout
		layout = QFormLayout()

		# adding rows
		# for name and adding input text
		layout.addRow(QLabel("Name"), self.nameLineEdit)

		# for degree and adding combo box
		layout.addRow(QLabel("Martial status"), self.marComboBox)

		# for age and adding spin box
		layout.addRow(QLabel("Age"), self.ageSpinBar)
		layout.addRow(QLabel("Say About You"), self.aboutEdit)
		layout.addRow(QLabel("Address"), self.addressEdit)
		layout.addRow(QLabel("DOB put in(/)"), self.DobEdit)
		layout.addRow(QLabel("MOBILE"), self.mobileEdit)
		# setting layout
		self.formGroupBox.setLayout(layout)
# main method
if __name__ == '__main__':

	# create pyqt5 app
	app = QApplication(sys.argv)

	# create the instance of our Window
	window = Window()

	# showing the window
	window.show()

	# start the app
	sys.exit(app.exec())


