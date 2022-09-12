import PySimpleGUI as sg
import cv2, sys, numpy, os
import os
#читаем конфиг
f = open('config.ini', 'r')
for line in f:
    l = [line.strip() for line in f]
f.close()
input1=f'"{l[0]}"'
input2=f'"{l[1]}"'
layout = [
    [sg.Text('Введите ФИО в формате: IvanovAP'), sg.InputText()
     ],
    [sg.Submit(), sg.Cancel()]
]
window = sg.Window('Выбор сотрудника для обучения', layout)
while True:                             # The Event Loop
    event, values = window.read()
    # print(event, values) #debug
    if event in (None, 'Exit', 'Cancel'):
        break
    if event == 'Submit':

        file1 = None
        if values[0]:
            file1 = values[0]
            if not file1 and file1 is not None:
                print('Неверный формат')
            print(file1)
            window.close()
            haar_file = 'haarcascade_frontalface_default.xml'
            datasets = 'datasets'
            try:
                os.mkdir(datasets)
            except:
                pass
            sub_data = file1
            path = os.path.join(datasets, sub_data)
            if not os.path.isdir(path):
                os.mkdir(path)
            (width, height) = (130, 100)
            face_cascade = cv2.CascadeClassifier(haar_file)
            # webcam = cv2.VideoCapture('rtsp://admin:Adm142!@@192.168.0.109/cam/realmonitor?channel=1&subtype=0')
            webcam = cv2.VideoCapture(0)
            count = 1
            while count < 25:
                (_, im) = webcam.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 3)
                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (width, height))
                    cv2.imwrite('% s/% s.png' % (path, count), face_resize)
                count += 1
                cv2.imshow(file1, im)
                key = cv2.waitKey(2)
                if key == 27:
                    break
