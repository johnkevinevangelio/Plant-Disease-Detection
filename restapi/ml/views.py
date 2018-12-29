from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect


from picamera import PiCamera
from time import sleep
from PIL import Image

from django.urls import reverse

def camera(request):

    camera = PiCamera()

    camera.rotation = 180

    camera.resolution = (1280, 720)
    camera.framerate = 24
    camera.start_preview(alpha=200)

    img = Image.open('/home/pi/Desktop/virtualenvs/PD/restapi/ml/overlay.png')
    pad = Image.new('RGBA', (
        ((img.size[0] + 31) // 32) * 32,
        ((img.size[1] + 15) // 16) * 16,
        ))

    pad.paste(img,(0, 0))

    o = camera.add_overlay(pad.tobytes(), size=img.size)

    o.alpha = 128
    o.layer = 3
    sleep(5)

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
        crop(image,(250, 130, 1050, 560),'/home/pi/Desktop/virtualenvs/PD/images/image%s.jpg' % i)

    camera.stop_preview()
    camera.close()
    
    url = reverse('start')
    return HttpResponseRedirect(url)
    



def start(request):
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = '/home/pi/Desktop/virtualenvs/PD/images'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = '/home/pi/Desktop/virtualenvs/PD/restapi/ml/PlantDiseaseDetection/healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

    
    print("hey")
    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt
    from serializer.models import Scan, Plant_Info

    last_scanned = Scan.objects.last()
    lscan_id = last_scanned.id
    print(lscan_id)

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'bacterial'
        elif np.argmax(model_out) == 2:
            str_label = 'viral'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'

        if str_label =='healthy':
            status ="HEALTHY"
            print(status)
            save = Plant_Info(scan_no=Scan(id=lscan_id), plant_no=num+1, plant_type="lettuce", condition=status, disease="None", diagnosis="You are a good planter")
            save.save()
        else:
            status = "UNHEALTHY"
            print(status)
           

        if str_label == 'bacterial':
            diseasename = "Bacterial Spot "
            print("Disease:"+ diseasename)
            save = Plant_Info(scan_no=Scan(id=lscan_id), plant_no=num+1, plant_type="lettuce", condition="unhealthy", disease=diseasename, diagnosis="You need to water the plants")
            save.save()
        elif str_label == 'viral':
            diseasename = "Yellow leaf curl virus "
            save = Plant_Info(scan_no=Scan(id=lscan_id), plant_no=num+1, plant_type="lettuce", condition="unhealthy", disease=diseasename, diagnosis="You need to water the plants")
            save.save()
        elif str_label == 'lateblight':
            diseasename = "Late Blight "
            print("Disease:"+ diseasename)
            save = Plant_Info(scan_no=Scan(id=lscan_id), plant_no=num+1, plant_type="lettuce", condition="unhealthy", disease=diseasename, diagnosis="You need to water the plants")
            save.save()
    return HttpResponse("It's done!")
