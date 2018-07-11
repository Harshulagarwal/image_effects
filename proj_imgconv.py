from Tkinter import *
import Tkinter as tk
from ttk import *
import cv2
import numpy as np
import tkMessageBox



root = Tk()
#root1=Tk()
tkMessageBox.showinfo(title="Download necessary files",message="Download haarcascade files and put them in /home/harshul/Documents/ directory")
tkMessageBox.showinfo(title="Click usage",message="Program will click your pic when u close the camera window if u hv clicked the button")

panedwindow = tk.PanedWindow(root, orient = VERTICAL)
panedwindow.pack(fill = BOTH, expand = False)

frame1 = tk.Frame(panedwindow, width = 100, height = 500, relief = SUNKEN)
frame2 = tk.Frame(panedwindow, width = 400, height = 1000, relief = SUNKEN)
panedwindow.add(frame1)
panedwindow.add(frame2)

button0=tk.Button(root,text="click")

def normal():


    cap=cv2.VideoCapture(0);

    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    while cap.isOpened():
      ret,frame3=cap.read()


      if(ret==True):
        cv2.imshow("Frame",frame3)
        '''button6=tk.Button(frame3,text="click")
        button6.pack()
        button6.config(justify=CENTER,font=("arial",20),command=click(frame3))
        '''
        button0.config(justify=CENTER,font=("arial",20),command=lambda:click(frame3))


        if cv2.waitKey(1) & 0xFF == 27:
          break

      # Break the loop
      else:
        break

    # When everything done, release the video capture and video write objects
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    # Closes all the frames


def grayscale():
    cap=cv2.VideoCapture(0);

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
      ret,frame=cap.read()

      if(ret==True):
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Frame",frame)
        button0.config(justify=CENTER,font=("arial",20),command=lambda:click(frame))

        if cv2.waitKey(1) & 0xFF == 27:
          break

      # Break the loop
      else:
        break

    # When everything done, release the video capture and video write objects
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    # Closes all the frames

def canny():
    cap=cv2.VideoCapture(0);

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
      ret,frame=cap.read()

      if(ret==True):
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame1=cv2.GaussianBlur(frame,(3,3),1)
        frame2=cv2.Canny(frame1,50,100,3)
        cv2.imshow("Frame",frame2)
        button0.config(justify=CENTER,font=("arial",20),command=lambda:click(frame2))

        if cv2.waitKey(1) & 0xFF == 27:
          break

      # Break the loop
      else:
        break

    # When everything done, release the video capture and video write objects
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    # Closes all the frames


def sketchi():
    cap=cv2.VideoCapture(0);

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
      ret,frame=cap.read()

      if(ret==True):
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frame1=cv2.GaussianBlur(frame,(3,3),1)
        frame2=cv2.Canny(frame,50,100,3)
        cv2.imshow("Frame",255-frame2)
        button0.config(justify=CENTER,font=("arial",20),command=lambda:click(frame2))

        if cv2.waitKey(1) & 0xFF == 27:
          break

      # Break the loop
      else:
        break

    # When everything done, release the video capture and video write objects
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    # Closes all the frames


def inv():
    cap=cv2.VideoCapture(0);

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
      ret,frame=cap.read()

      if(ret==True):
        frame=255-frame
        cv2.imshow("Frame",frame)
        button0.config(justify=CENTER,font=("arial",20),command=lambda:click(frame))

        if cv2.waitKey(1) & 0xFF == 27:
          break

      # Break the loop
      else:
        break

    # When everything done, release the video capture and video write objects
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    # Closes all the frames



def faced():
    face=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_frontalface_default.xml')
    #eye=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_eye.xml')
    #smile=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_smile.xml')

    cap=cv2.VideoCapture(0)

    while True:
        ret,img=cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        f=face.detectMultiScale(gray, 1.3,4)
        for (x,y,w,h) in f:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)


        cv2.imshow('img',img)
        button0.config(justify=CENTER,font=("arial",20),command=lambda:click(img))

        if cv2.waitKey(1) & 0xFF == 27  :

            break

    cap.release()
    cv2.destroyAllWindows()



def eyed():

        face=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_frontalface_default.xml')
        eye=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_eye.xml')
        #smile=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_smile.xml')

        cap=cv2.VideoCapture(0)

        while True:
            c=0
            ret,img=cap.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            f=face.detectMultiScale(gray, 1.3,4)
            for (x,y,w,h) in f:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

                eyeregion=gray[y:y+h , x:x+w]
                e=eye.detectMultiScale(eyeregion)

                for (ex,ey,ew,eh) in e:
                    c=c+1
                    if c<3:
                        cv2.rectangle(img, (ex+x,ey+y), (ex+x+ew,ey+y+eh), (0,255,0), 2)
                    else:
                        break



            cv2.imshow('img',img)
            button0.config(justify=CENTER,font=("arial",20),command=lambda:click(img))

            if cv2.waitKey(1) & 0xFF == 27  :

                break

        cap.release()
        cv2.destroyAllWindows()

def filteri():
    face=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_frontalface_default.xml')
    eye=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_eye.xml')
    #smile=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_smile.xml')
    cap=cv2.VideoCapture(0)

    while True:
        c=0
        d=0
        img2=cv2.imread("/home/harshul/Documents/catt.jpg")
        ret,img=cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask)


        f=face.detectMultiScale(gray, 1.3, 4)

        #print f

        for (x,y,w,h) in f:
            x_offset=x
            y_offset=y-100

            roi = img[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1] ]
            img21 = cv2.bitwise_and(roi,roi,mask = mask_inv)
            #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

            dst = cv2.add(img21,img2_fg)

            img[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]]=dst

        cv2.imshow('img',img)
        button0.config(justify=CENTER,font=("arial",20),command=lambda:click(img))

        #cv2.imshow("img1",dst)
        if cv2.waitKey(1) & 0xFF == 27  :

            break

    cap.release()
    cv2.destroyAllWindows()


def bigeye():
    face=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_frontalface_default.xml')
    eye=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_eye.xml')
    #smile=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_smile.xml')

    cap=cv2.VideoCapture(0)

    while True:
        c=0
        eye2=np.array([0,0,0,0])
        eye4=np.array([0,0,0,0])
        ret,img=cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        f=face.detectMultiScale(gray, 1.3,4)
        for (x,y,w,h) in f:
            #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

            eyeregion=gray[y:y+h , x:x+w]
            e=eye.detectMultiScale(eyeregion)
            img=cv2.GaussianBlur(img,(3,3),3)
            for (ex,ey,ew,eh) in e:
                c=c+1
                if c<=2:

                    eye1=img[ey+y:ey+y+eh,ex+x:ex+x+ew]
                    eye2=cv2.resize(eye1,(2*ew-10,2*eh-10),interpolation=cv2.INTER_AREA)
                    eye2=cv2.GaussianBlur(eye2,(3,3),1)
                    img[ey+y-eh+30:ey+y+eh+20,ex+x-ew+35:ex+x+ew+25]=eye2
                    #cv2.rectangle(img, (ex+x,ey+y), (ex+x+ew,ey+y+eh), (0,255,0), 2)
                    img=cv2.GaussianBlur(img,(5,5),3)

                else:
                    break

        cv2.imshow("img",img)

        #cv2.imshow("img1 ",eye4)

        if cv2.waitKey(1) & 0xFF == 27  :

            break

    cap.release()
    cv2.destroyAllWindows()

def thug():
    import cv2
    import numpy as np

    face=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_frontalface_default.xml')
    eye=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_righteye_2splits.xml')
    #smile=cv2.CascadeClassifier('/home/harshul/Documents/haarcascade_smile.xml')
    cap=cv2.VideoCapture(0)

    while True:
        c=0
        d=0
        img2=cv2.imread("/home/harshul/Documents/thug.jpg")
        img2 = cv2.bitwise_not(img2)

        ret,img=cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask)


        f=face.detectMultiScale(gray, 1.3, 4)


        for (x,y,w,h) in f:

            eyeregion=gray[y:y+h , x:x+w]
            e=eye.detectMultiScale(eyeregion)

            for (ex,ey,ew,eh) in e:
                c=c+1
                if c<2 :
                    x_offset=ex+x-50
                    y_offset=ey+y+18

                    roi = img[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1] ]
                    img21 = cv2.bitwise_and(roi,roi,mask = mask_inv)
                    img21 = cv2.bitwise_not(img21)

                    #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

                    dst = cv2.add(img21,img2_fg)
                    dst = cv2.bitwise_not(dst)
                    img[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]]=dst


                    #cv2.rectangle(img, (ex+x,ey+y), (ex+x+ew,ey+y+eh), (255,255,0), 2)
                else:
                    break
        cv2.imshow('img',img)
        #cv2.imshow("img1",dst)

        if cv2.waitKey(1) & 0xFF == 27  :

            break


        if cv2.waitKey(1) & 0XFF == ord('s'):
            tkMessageBox.showinfo(title="saved",message="pic saved")
            cv2.imwrite("/home/harshul/Desktop/img.jpg",img)


    cap.release()
    cv2.destroyAllWindows()


label=tk.Label(frame1,text="choose one of the effects below:")
label.pack()
label.config(justify=CENTER,font=("arial",20,"bold"),padx=10,pady=10,background="yellow",foreground="blue")

button=tk.Button(frame2,text="normal")
button.pack()
button.config(justify=CENTER,font=("arial",20),command=normal ,width=20)

button1=tk.Button(frame2,text="grayscale")
button1.pack()
button1.config(justify=CENTER,font=("arial",20),command=grayscale,width=20)


button2=tk.Button(frame2,text="canny edge")
button2.pack()
button2.config(justify=CENTER,font=("arial",20),command=canny,width=20)


button5=tk.Button(frame2,text="sketch")
button5.pack()
button5.config(justify=CENTER,font=("arial",20),command=sketchi,width=20)

button3=tk.Button(frame2,text="negative")
button3.pack()
button3.config(justify=CENTER,font=("arial",20),command=inv,width=20)

det=StringVar()
cbox=Combobox(frame2,text="detector",textvariable=det,state="readonly", width=20)
cbox.pack()
cbox.config(justify=CENTER,font=("arial",20),value=('face detector','face and eye detector'))
cbox.set("choose detector")
cbox.bind("<<ComboboxSelected>>",lambda e:a())

button4=tk.Button(frame2,text="detect")
button4.pack()
button4.config(justify=CENTER,font=("arial",20), width=15)


button7=tk.Button(frame2,text="snap filter",width=20)
button7.pack()
button7.config(justify=CENTER ,font=("arial",20),command=filteri, width=20)


button8=tk.Button(frame2,text="big eye")
button8.pack()
button8.config(justify=CENTER ,font=("arial",20),command=bigeye, width=20)

button9=tk.Button(frame2,text="thug life")
button9.pack()
button9.config(justify=CENTER ,font=("arial",20),command=thug, width=20)


def click(frame2):
    #cap=cv2.VideoCapture(0)

    #while cap.isOpened():
      #ret,frame=cap.read()
    cv2.imwrite('img2.jpg',frame2)


def a():
    if cbox.get()=="face detector":
        button4.config(justify=CENTER,font=("arial",20),command=faced, width=15)
    if cbox.get()=="face and eye detector":
        button4.config(justify=CENTER,font=("arial",20),command=eyed, width=15)
root.mainloop()
