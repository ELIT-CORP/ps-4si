#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

#Trata a imagem do frame
def frameProcessing(frame):
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(frame_gray, (5,5), 0)

        threshold = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]

        return threshold 

cap = cv2.VideoCapture("q1A.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:

        print("Can't receive frame (stream end?). Exiting ...")
        break

    else:
        frameProcessed = frameProcessing(frame)

        contornos, _ = cv2.findContours(frameProcessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contornos_ordenados = sorted(contornos, key = cv2.contourArea, reverse=True)

        # Desenha o retangulo verde em volta do objeto de maior massa
        xb,yb,wb,hb = cv2.boundingRect(contornos_ordenados[0])
        cv2.rectangle(frame,(xb,yb),(xb+wb,yb+hb),(0,255,0),2) 

        if len(contornos) == 2:
            frame = cv2.putText(frame, 'Numero de objetos encontrados: ' + str(len(contornos)), (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Sem colisao', (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            xl,yl,wl,hl = cv2.boundingRect(contornos_ordenados[1])      

            if (xb + wb) < xl:
                frame = cv2.putText(frame, 'Barreira ultrapassada', (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

        else:
            frame = cv2.putText(frame, "COLISAO!! COLISAO! 'O'", (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5, cv2.LINE_AA) 

        # Exibe resultado
        cv2.imshow("Feed", frame)

        # Wait for key 'ESC' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

# That's how you exit
cap.release()
cv2.destroyAllWindows()