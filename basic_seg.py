import cv2
import numpy as np

min_area = 300
max_area = 5000

#1) Le a foto e aplica um efeito de borramento (passa-baixa)
img = cv2.imread('PEDRAS_AVIAO_SSS - Copia.png', cv2.IMREAD_GRAYSCALE)
img_after_lp = cv2.GaussianBlur(img, (5, 5), 0)

#2) Encontra as bordas presentes na imagem
edges_from_canny = cv2.Canny(img_after_lp, threshold1=40, threshold2=150)

#3) Encontra todos os contornos
contours_edges, _ = cv2.findContours(edges_from_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_edges = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#3) Itera nos contornos e filtra pelo tamanho da area
for countour in contours_edges:
    area = cv2.contourArea(countour)
    if min_area < area < max_area:
        x, y, w, h = cv2.boundingRect(countour)
        cv2.rectangle(output_edges, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output_edges, f'Objeto> E:{int(area)}', (x, max(y + 15, 15)),  # y + 15 coloca o texto dentro
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


cv2.imshow("Imagem original", img)
cv2.imshow("Depois do passa baixa", img_after_lp)
cv2.imshow("Contornos via Canny", output_edges)
cv2.imshow("Canny edges (bordas)", edges_from_canny)

cv2.imwrite("img_result.png", output_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
