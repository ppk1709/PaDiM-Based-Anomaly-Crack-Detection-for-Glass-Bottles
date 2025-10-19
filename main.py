# main.py
import tkinter as tk
from inspector_app import InspectorApp

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from tkinter import ttk
        style = ttk.Style(root)
        style.theme_use("clam")
    except:
        pass
    app = InspectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_quit)
    root.mainloop()