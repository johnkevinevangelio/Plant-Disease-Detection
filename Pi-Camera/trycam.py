from picamera import PiCamera
from time import sleep
from PIL import Image


camera = PiCamera()


camera.rotation = 180

camera.resolution = (1280, 720)
camera.framerate = 24
camera.start_preview()

img = Image.open('overlay.png')
pad = Image.new('RGBA', (
    ((img.size[0] + 31) // 32) * 32,
    ((img.size[1] + 15) // 16) * 16,
    ))

pad.paste(img,(0, 0))

o = camera.add_overlay(pad.tobytes(), size=img.size)

o.alpha = 128
o.layer = 3
for i in range(5):
    sleep(5)
    camera.capture('/home/pi/Desktop/virtualenvs/PD/images/image%s.jpg' % i)
camera.stop_preview()


