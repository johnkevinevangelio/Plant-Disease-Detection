from picamera import PiCamera
from time import sleep
from PIL import Image
import numpy as np




camera = PiCamera()


camera.rotation = 180

camera.resolution = (1280, 720)
camera.framerate = 24
camera.start_preview(alpha=150)

img = Image.open('overlay2.png')
pad = Image.new('RGBA', (
    ((img.size[0] + 31) // 32) * 32,
    ((img.size[1] + 15) // 16) * 16,
    ))

pad.paste(img,(0, 0))

o = camera.add_overlay(pad.tobytes(), size=img.size)

o.alpha = 255
o.layer = 3


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()


for i in range(2):
    sleep(5)
    camera.capture('/home/pi/Desktop/virtualenvs/PD/images/image%s.jpg' % i)
    image = '/home/pi/Desktop/virtualenvs/PD/images/image%s.jpg' % i
    crop(image,(460, 150, 830, 580),'/home/pi/Desktop/virtualenvs/PD/images/image%s.jpg' % i)
camera.stop_preview()
camera.close()



