import cv2
import numpy as np
from scipy.spatial import distance as dist

cap = cv2.VideoCapture("d:/Estudos/Python/ps-4si/q2/q2.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # # Seu código aqui.......
    resize = cv2.resize(frame, (1280, 720))
    orig = resize.copy()
    vermelhoContador = 0
    pretoContador = 0

    edges = cv2.Canny(resize, 100, 500)
    cnts, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts,
                                            [cv2.boundingRect(c) for c in cnts]), key=lambda b: b[1][0], reverse=False))

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = order_points(box)
        x = int(box[0][0]) + 20
        y = int(box[0][1]) + 58
        color = orig[y, x]
        
        if (color <= [8, 8, 8]).all():
            pretoContador += 1
        else:
            vermelhoContador += 1
            
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    cv2.putText(orig, "Vermelho: " + str(vermelhoContador),(10, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(orig, "Preto: " + str(pretoContador),(10, 60), font, 1, (0, 0, 0), 2)

    # see the results
    cv2.imshow('Feed', orig)
    if cv2.waitKey(20) == ord('q'):
        break

cv2.destroyAllWindows()
# That's how you exit
cap.release()
cv2.destroyAllWindows()
