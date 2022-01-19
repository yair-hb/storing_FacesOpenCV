from email.mime import image
import cv2
import os 

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
imagen = cv2.imread('oficina.png')
imagenAux = imagen.copy()
imGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

rostros = faceClassif.detectMultiScale(imGris,1.1,5)

contador = 0

for (x,y,w,h) in rostros:
    cv2.rectangle(imagen, (x,y), (x+w,y+h),(128,0,255),2)
    caras = imagenAux[y:y+h,x:x+w]
    caras = cv2.resize(caras,(150,150), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('rostros_{}.jpg'.format(contador),caras)
    contador = contador +1

    cv2.imshow('rostro',caras)
    cv2.imshow('imagen',imagen)
    cv2.waitKey(0)
cv2.destroyAllWindows()