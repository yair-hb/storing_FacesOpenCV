import cv2
import os 

rutaImagenes = r"C:\Users\gabri\OneDrive\Escritorio\YAIR\Storing_FacesOpenCV\imagenes"
listaRutaImag = os.listdir(rutaImagenes)

if not os.path.exists('Rostros encontrados'):
    print ('Carpeta creada: Rostros Encontrados')
    os.makedirs('Rostros encontrados')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

contador = 0
for nombreImagen in  listaRutaImag:
    imagen = cv2.imread(rutaImagenes+'/'+nombreImagen)
    imagenAux = imagen.copy()
    imGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    rostros = faceClassif.detectMultiScale(imGris,1.2,5)

    for (x,y,w,h) in rostros:
        cv2.rectangle(imagen, (x,y),(x+w,y+h),(128,0,255),2)
    cv2.rectangle(imagen, (10,5),(520,25),(255,255,255),-1)
    cv2.putText(imagen,'Presiona la tecla S para almacenar los rostros encontrados',(10,20),2,0.5,(128,0,255),1,cv2.LINE_AA)
    cv2.imshow('imagen', imagen)
    k = cv2.waitKey(0)
    if k == ord('s'):
        for (x,y,w,h,) in rostros:
            caras = imagenAux[y:y+h,x:x+w]
            caras = cv2.resize(caras, (250,250),interpolation=cv2.INTER_CUBIC)
            cv2.imshow('Rostro', caras)
            cv2.waitKey(0)
            cv2.imwrite('Rostros encontrados/rostro_{}.jpg'.format(contador),caras)
            contador = contador +1
    elif k ==27:
        break
cv2.destroyAllWindows()