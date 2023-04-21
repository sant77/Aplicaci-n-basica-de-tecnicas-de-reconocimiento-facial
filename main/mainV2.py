
"""
Nombre: Santiago Duque Ramos
fecha: 22/12/2022

"""

# Importamos librerias
from tkinter import *
from PIL import Image, ImageTk
import cv2
import imutils
import pathlib
import os
import errno
import functions_face as ff
from tkinter import messagebox
import time

# Se localiza el path donde esta el archivo

path = str(pathlib.Path(__file__).parent.absolute())

pathActual = path[0:len(path)-4]

faceClassif = cv2.CascadeClassifier(path + "\haarcascade_frontalface_default.xml")

train = False

# Lista de usuarios registrados
list_people = os.listdir(pathActual+"/data")
 
# Funcion Visualizar
def visualizar():

    # Parametros del texto en el video
    texto = "None"
    ubicacion = (100,100)
    font = cv2.FONT_HERSHEY_TRIPLEX
    tamañoLetra = 1
    colorLetra = (221,82,196)
    grosorLetra = 2

    global pantalla, frame

    # Leemos la videocaptura
    if cap is not None:
        ret, frame = cap.read()

        # Si es correcta
        if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Activar la deteción
            faces = faceClassif.detectMultiScale(gray, 1.3, 5)
            
            # Dibujar el rectangulo donde está la cara
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

            # Si ya se entrenó algun modelo
            if train == True:

                # Si se detecta alguna cara
                
                if len(faces) != 0  :
                    for (x,y,w,h) in faces:
                        #cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                        pass

                    # Recortar la cara
                    imageOut = frame[y:y+h,x:x+w]

                    # aplicar LBP
                    lbp_hist_test = ff.lbpImage(imageOut)
                    
                    # Aplicar el clasificador
                    y_pred = model.predict(lbp_hist_test)

                    # Colocar el nombre
                    
                    texto = str(y_pred[0])
            


            #Escribir texto
            cv2.putText(frame, texto, ubicacion, font, tamañoLetra, colorLetra, grosorLetra)

            frameShow = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frameShow = imutils.resize(frameShow, width=640)

            # Convertimos el video
            im = Image.fromarray(frameShow)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)

        else:
            cap.release()


# Funcion iniciar
def iniciar():
    global cap
    # Elegimos la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    visualizar()
    

# Funcion finalizar

def finalizar():

    cap.release()
    cv2.DestroyAllWindows()

# Guardar nuevo sujeto


def save_input():
    global input_text
    input_text = input_entry.get()
    
    if input_text in list_people:

         messagebox.showinfo("¡Error!", "Usuario ya registrado")

    else:
        i = 0
        crearDirectorio(input_text)
        cap = cv2.VideoCapture(0)

        messagebox.showinfo("Tomando sus datos", "Por favor mueva su rostro ligeramente de izquierda a derecha")
        # Se toma 20 fotos en la clasificación

        while i < 20:

            ret,frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            time.sleep(0.4)

            if len(faces) != 0  :

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                
                imageOut = frame[y:y+h,x:x+w]
                cv2.imwrite(pathActual +"/data"+"/"+input_text+"/"+str(i)+".jpg",imageOut)

                i += 1

            

            cv2.imshow('frame',frame)

        cap.release()
        cv2.destroyAllWindows()     
        nueva_ventana.destroy()

# Solo se activa el boton si guarda

def validate_input(text):
    if len(text) == 0:
        input_entry.config(bg="red")
        save_button.config(state="disabled")
    else:
        input_entry.config(bg="white")
        save_button.config(state="normal")
    return True


def crearSujeto():
    global nueva_ventana, input_entry, save_button, input_text
    # Crea la nueva ventana
    
    nueva_ventana = Toplevel()
    nueva_ventana.title("Crear nuevo usuario")
    nueva_ventana.geometry("200x200")
    
    input_label = Label(nueva_ventana, text="Ingrese su nombre:")
    input_label.pack()

    validate_cmd = nueva_ventana.register(validate_input)
    input_entry = Entry(nueva_ventana, validate="key", validatecommand=(validate_cmd, "%P"))
    input_entry.pack()

    save_button = Button(nueva_ventana, text="Guardar", state="disabled", command=save_input)
    save_button.pack()


# Crear nuevo directorio, donde se guarda las imagenes

def crearDirectorio(text):

    try:
        os.mkdir(pathActual +"/data"+"/"+text)

    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def entrenarModelo():

    global train, model

    # Entrenar modelo knn

    if selected_option == "LBP + KNN" and len(list_people)>1:
        
        model = ff.trainModel()
        
        train = True
        
        messagebox.showinfo("¡Entrenado!", f"Entrenamiento de {selected_option} completado.")
        #print("Entrenado...")
    
    elif selected_option == "LBP + SVM" and len(list_people)>1: 
        model = ff.trainModel("svm")
        
        train = True
        
        messagebox.showinfo("¡Entrenado!", f"Entrenamiento de {selected_option} completado.")

    elif selected_option == "LBP + Gauss" and len(list_people)>1:
        
        model = ff.trainModel("gauss")
        
        train = True
        
        messagebox.showinfo("¡Entrenado!", f"Entrenamiento de {selected_option} completado.")
    
    else:
        messagebox.showinfo("Error!", "Por favor revise el numero de personas registradas o seleccione un modelo")

# Seleccionar otros modelos...... proximamente
def on_select(val):
    global selected_option
    selected_option = val


#  Ventana Principal
# Pantalla
pantalla = Tk()
pantalla.title("Prototipo de reconocimiento facial")
pantalla.geometry("1280x720")  # Asignamos la dimension de la ventana

# Fondo
imagenF = PhotoImage(file= pathActual + "img/Fondo.png")
background = Label(image = imagenF, text = "Fondo")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)

# Interfaz
texto1 = Label(pantalla, text="CAPTURA DE VIDEO",font='Helvetica 18 bold')
texto1.place(x = 700, y = 50)

texto2 = Label(pantalla, text="Menú",font='Helvetica 18 bold')
texto2.place(x = 190, y = 50)


# Botones
# Iniciar Video
imagenBI = PhotoImage(file= pathActual +"img/InicioV2.png")
inicio = Button(pantalla, text="Iniciar", image=imagenBI, height="40", width="200", command=iniciar)
#inicio = Button(pantalla, text="Iniciar", command=iniciar)
inicio.place(x = 130, y = 150)

# Finalizar Video
imagenBF = PhotoImage(file= pathActual + "img/ReinicioV2.png")
fin = Button(pantalla, text="Finalizar", image= imagenBF, height="40", width="200", command=finalizar)
#fin = Button(pantalla, text="Finalizar", command=finalizar)
fin.place(x = 130, y = 250)


#Guardar nuevo sujeto
imagenGenerar = PhotoImage(file= pathActual + "img/generarV2.png")
giro = Button(pantalla, text="Giro", image=imagenGenerar, height="40", width="200",command=crearSujeto)
giro.place(x = 130, y = 350)

#Entrenar modelo
imagenEntrenar = PhotoImage(file= pathActual + "img/muV2.png")
entrenar = Button(pantalla, text="Giro", image=imagenEntrenar, height="40", width="200",command=entrenarModelo)
entrenar.place(x = 130, y = 450)

#Seleccionar modelo

# Definir opciones
options = ["LBP + KNN","LBP + SVM","LBP + Gauss"]

# Variable para almacenar la opción seleccionada
selected_option = StringVar()

# Crear menú desplegable y agregar opciones
dropdown = OptionMenu(pantalla,selected_option, *options, command=on_select)

# Decorar menú desplegable
dropdown.config(bg="#333333", fg="#FFFFFF", activebackground="#444444", activeforeground="#FFFFFF", highlightthickness=0)

dropdown.place(x = 180, y = 550)


# Video
lblVideo = Label(pantalla)
lblVideo.place(x = 500, y = 100)

lblVideo2 = Label(pantalla)
lblVideo2.place(x = 470, y = 500)

pantalla.mainloop()


