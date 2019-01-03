import serial
import time


ser = serial.Serial('/dev/ttyACM1', 9600)
time.sleep(5)
k=True
while(k):
    ser.write(str.encode('1'))
    time.sleep(2)
    ser.write(str.encode('0'))
