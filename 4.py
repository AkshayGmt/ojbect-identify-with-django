import cv2
import cv2
import numpy as np
import webcolors
import pandas as pd
img = cv2.imread('bus.jpg')

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=2)

    text = "{}:{:.1f}".format(classes[classId], score)
    print(text)
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 255, 0), thickness=2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
r = g = b = xpos = ypos = 0

#Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


val = input("Please Enter Your questions: ")


if val=="colors":
    #print("color detect")
    
    # Define a function to get the name of the closest color in the webcolors database
    def getColorName(rgb):
        minimum = 10000
        for i in range(len(csv)):
            d = abs(R- int(csv.loc[i,"r"])) + abs(G- int(csv.loc[i,"g"]))+ abs(B- int(csv.loc[i,"b"]))
            if(d<=minimum):
                minimum = d
                cname = csv.loc[i,"color_name"]
        return cname

    # Get the dominant color of the img
    pixels = np.float32(img.reshape(-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    # Get the name of the closest color in the webcolors database
    #color_name = getColorName(dominant)
    color_name = getColorName(dominant) #+ ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)

    print(color_name)
    # Show the dominant color and its name
    print(f"Dominant color: RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}")


else:
    print("")
