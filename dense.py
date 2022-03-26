import cv2
import numpy as np




#cap = cv2.VideoCapture('D:\\openpose_data\\result_ex1_blending.avi')
cap = cv2.VideoCapture('D:\\openpose_data\\fall_down_mixamo\\fall_down_mixamo\\fall_down026.avi')
#cap = cv2.VideoCapture('D:\\python_project\\st-gcn\\fall_train.avi')
#cap = cv2.VideoCapture('D:\\openpose_data\\fall_down_mixamo\\fall_down_mixamo\\fall_train.avi')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
print(hsv)
hsv[..., 1] = 255

while (1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb)
    if cv2.waitKey(60) > 0: break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()