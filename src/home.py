from tkinter import *
from main import *
from PIL import ImageTk,Image

window = Tk()
bg_photo=ImageTk.PhotoImage(Image.open('../pic/FirstPage.png'))
icon=PhotoImage(file='../pic/logospkt.png')
width_value=window.winfo_screenwidth()
height_value=window.winfo_screenheight()
window.geometry("%dx%d+0+0"%(width_value,height_value))
window.title('Group 7 - Face Attendance System')
window.iconphoto(True,icon)

bg=Label(image=bg_photo)
bg.pack()
newatt=Button(window,
                  text='Home',
                  command=mainfunc,
                  font=('Montserrat',15),
                  height='2',
                  width='15',
                  fg='#ffffff',
                  bg='#3700b3',

                  )
newatt.place(x=830,y=800)
window.mainloop()