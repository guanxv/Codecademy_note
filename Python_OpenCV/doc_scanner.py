import cv2
import numpy as np
from stackimg import stackImages

frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)


def preProcessing(img):
    imgResize = cv2.resize(img, (640, 480))
    imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 10)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    imgStack = stackImages(
        0.8, [[imgResize, imgGray, imgBlur], [imgCanny, imgDial, imgThres]]
    )

    cv2.imshow("Img Preprocessing", imgStack)

    return imgThres


def getContours(img):

    biggest = np.array([])
    maxArea = 0

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    x, y, w, h = (0, 0, 0, 0)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 2000:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if area > maxArea and len(approx) == 4:

                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 10)
    return biggest


def reorder(myPoints):

    myPointsNew = np.zeros((4, 1, 2), np.int32)

    if myPoints.size != 0:
        myPoints = myPoints.reshape((4, 2))

        add = myPoints.sum(1)

        print("add", add)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]

        diff = np.diff(myPoints, axis=1)

        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]

        print("NewPoints", myPointsNew)

    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)

    width = 1280
    height = 720

    print(biggest)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpped = cv2.warpPerspective(img, matrix, (width, height))

    imgCropped = imgWarpped[
        20 : imgWarpped.shape[0] - 20, 20 : imgWarpped.shape[1] - 20
    ]
    imgCropped = cv2.resize(imgCropped, (width, height))

    return imgCropped


while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgThres = preProcessing(img)
    biggest = getContours(imgThres)

    print(biggest.shape)
    imgWarpped = getWarp(img, biggest)

    imgArray = ([img, imgThres], [imgContour, imgWarpped])

    stackedImg = stackImages(0.8, imgArray)

    cv2.imshow("Result", imgContour)
    cv2.imshow("workflow", stackedImg)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

