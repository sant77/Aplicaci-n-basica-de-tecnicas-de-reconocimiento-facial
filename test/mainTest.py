"""
Nombre: Santiago Duque Ramos
fecha: 22/12/2022

"""

# Importamos librerias
from tkinter import *
from PIL import Image, ImageTk
import tkinter.messagebox
import customtkinter as ctk

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Funcion Visualizar
def visualizar():
    pass
# Funcion iniciar
def iniciar():
    pass
# Funcion finalizar
def finalizar():
    pass
def save_input():
    pass
def validate_input(text):
    pass

def crearSujeto():

    nueva_ventana = ctk.CTkToplevel()
    nueva_ventana.title("Crear nuevo usuario")
    nueva_ventana.geometry("300x540")
    
def crearDirectorio(text):
    pass


def entrenarModelo():
    pass


#  Ventana Principal
# Pantalla
pantalla = ctk.CTk()
pantalla.title("Prototipo extraccion de cedula")
pantalla.geometry("1280x720")  # Asignamos la dimension de la ventana



# Interfaz
texto1 = ctk.CTkLabel(pantalla, text="CAPTURA DE VIDEO",font='Helvetica 18 bold')

texto1.place(x = 700, y = 50)


# Botones
# Iniciar Video
#imagenBI = PhotoImage(file= pathActual +"img/InicioV2.png")
inicio = ctk.CTkButton(pantalla,text="Iniciar reconocimiento" ,command=iniciar)
#inicio = Button(pantalla, text="Iniciar", command=iniciar)
inicio.place(x = 130, y = 200)

# Finalizar Video
#imagenBF = PhotoImage(file= pathActual + "img/ReinicioV2.png")
fin = ctk.CTkButton(pantalla,text="Parar reconocimiento", command=finalizar)
#fin = Button(pantalla, text="Finalizar", command=finalizar)
fin.place(x = 130, y = 300)


#Generar sujeto
#imagenGenerar = PhotoImage(file= pathActual + "img/generarV2.png")
giro = ctk.CTkButton(pantalla, text="Crear usuario", command=crearSujeto)
giro.place(x = 130, y = 400)

#Generar sujeto
#imagenEntrenar = PhotoImage(file= pathActual + "img/generarV2.png")
entrenar = ctk.CTkButton(pantalla,text="¡Entrenar modelo!",command=entrenarModelo)
entrenar.place(x = 130, y = 500)
  

# Video
lblVideo = ctk.CTkLabel(pantalla)
lblVideo.place(x = 500, y = 100)

lblVideo2 = ctk.CTkLabel(pantalla)
lblVideo2.place(x = 470, y = 500)

pantalla.mainloop()


