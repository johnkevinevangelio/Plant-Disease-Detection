import sys
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm

def main(argv):

    #initialize the colormap
    colormap = mpl.cm.Spectral
    cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
    scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)


    while (True):
        # Capture frame-by-frame
        image = cv2.imread('/home/pi/Desktop/ml/PlantDiseaseDetection/testpicture/lettuce_septoria_blight.jpg')
        # Our operations on the frame come here
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#add blur to make it more realistic
        blur = cv2.GaussianBlur(gray,(15,15),0)
		#assign colormap
        colors = scalarMap.to_rgba(blur, bytes=False)

        # Display the resulting frame
        cv2.imshow('frame', colors)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
