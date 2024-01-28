import cv2 as cv
import os
import easyocr


reader = easyocr.Reader(['en'])
harcascade = r"C:\Users\niket\OneDrive\Desktop\roughtProjects\numberPlateDetection\haarcascade_russian_plate_number.xml"

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

min_area = 500
count =0

while True:
    istrue,img =cap.read()

    plate_cas= cv.CascadeClassifier(harcascade)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    plates = plate_cas.detectMultiScale(img_gray,1.1,4)

    for(x,y,w,h) in plates:
        area= w*h

        if(area>min_area):
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            cv.putText(img,"numberPlate",(x,y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),thickness=2)

            img_roi = img[y:y+h,x:x+w]
            cv.imwrite(os.path.join('platesOutput', 'outputImg_{}.jpg'.format(count)), img_roi)
            cv.imshow('ROI',img)
            output = reader.readtext(img_roi) 
            print(output)
            
            if output:
                path= os.path.join('platesNum','plateNumOut.txt')
                
                with open(path,'a') as file:
                    for result in output:
                        if(len(result)>1):
                            file.write(output[0][1] +'\n')

                count = count +1
            
            # cv.imshow('ROI',img)
    cv.imshow('result',img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    