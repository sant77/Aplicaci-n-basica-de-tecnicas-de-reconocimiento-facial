import tkinter as tk

def on_select(val):
    global selected_option
    selected_option = val

root = tk.Tk()
root.title("Menú desplegable")

# Definir opciones
options = ["Opción 1", "Opción 2", "Opción 3"]

# Variable para almacenar la opción seleccionada
selected_option = tk.StringVar(value=options[0])  # Valor inicial: "Opción 1"

# Crear menú desplegable y agregar opciones
dropdown = tk.OptionMenu(root, selected_option, *options, command=on_select)

# Decorar menú desplegable
dropdown.config(bg="#333333", fg="#FFFFFF", activebackground="#444444", activeforeground="#FFFFFF", highlightthickness=0)

dropdown.pack()

root.mainloop()