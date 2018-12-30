import sys
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm

def main(argv):

    #initialize the colormap
    colormap = mpl.cm.jet
    cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
    scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)


    # Capture frame-by-frame
    image = cv2.imread('/home/pi/Desktop/virtualenvs/PD/PlantDiseaseDetection/testpicture/lettuce_septoria_blight.jpg')
    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
            #add blur to make it more realistic
    blur = cv2.GaussianBlur(gray,(15,15),0)
            #assign colormap
    colors = scalarMap.to_rgba(blur, bytes=True)

    # Display the resulting frame
    
    cv2.imshow('frame', colors)
    cv2.imwrite("/home/pi/Desktop/virtualenvs/PD/convertimage/image.jpg",colors)
    
    cv2.waitKey(0)
    # When everything done, release the capture
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
