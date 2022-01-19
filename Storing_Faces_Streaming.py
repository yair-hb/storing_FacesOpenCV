from ast import While
import cv2
import os

if not os.path.exists('Rostros encontrados'):
    print ('carpeta Creada: Rostros encontrados')
    os.makedirs('Rostros encontrados')

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

contador = 0
while True:
    ret, frame, captura.read()
    frame = cv2.flip(frame,1)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameAux = frame.copy()

    faces = faceClassif.detectMultiScale(gris, 1.3,5)

    k = cv2.waitKey(1)
    if k == 27:
        break

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(128,0,255),2)
        rostro = frameAux[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        if k == ord ('s'):
            cv2.imwrite('Rostro encontrado/rostro_{}.jpg'.format(contador))
            cv2.imshow('rostro', rostro)
            contador = contador +1
    
    cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
    cv2.putText(frame, 'Presione tecla S para almacenar rostros encontrados',(10,20),2,0.5,(128,0,255),1,cv2.LINE_AA)
    cv2.imshow('frame', frame)
captura.release()
cv2.destroyAllWindows()