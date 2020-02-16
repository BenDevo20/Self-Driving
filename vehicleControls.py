#import RPi.GPIO as gpio
import sys
import time
import tkinter as tk

#def init():
 #   gpio.setmode(gpio.BOARD)
  #  #Change these numbers in accordance with pins
   # gpio.setup(1111, gpio.OUT)
    #gpio.setup(1111, gpio.OUT)
    #gpio.setup(1111, gpio.OUT)
    #gpio.setup(1111, gpio.OUT)


#def forward():
    #Change these in accordance with pins
    #2/4 wheel drive
  #  gpio.output(1111, True)
  #  gpio.output(1111, True)
  #  gpio.output(1111, False)
  #  gpio.output(1111, False)
  #  gpio.cleanup()

#def reverse():


#def turnRight():

#def turnLeft():

def key_input(event):
    #init()
    print('Key:', event.char)
    key_press = event.char
    sleep_time = 0.050

    #Change to arrow Keys
    if key_press.lower() == 'w':
        print('Forward')
        #forward(sleep_time)
    elif key_press.lower() == 's':
        print('Reverse')
        #reverse(sleep_time)
    elif key_press.lower() == 'a':
        print('Left')
        #turnLeft(sleep_time)
    elif key_press.lower() == 'd':
        print('Right')
        #turnRight(sleep_time)

root = tk.Tk()
root.bind('<KeyPress>', key_input)
root.mainloop()





