import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загружаем цветное изображение
img = cv2.imread('winter_cat.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f'Размер изображения = {img_rgb.shape}')
print(f'Диапазон яркостей = от {np.min(img_rgb)} до {np.max(img_rgb)}')
hist = np.zeros(256, dtype = int)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img_gray.shape
for y in range(h):
    for x in range(w):
        yark = img_gray[y, x]
        hist[yark] = hist[yark] + 1
print(f'Сумма всех пикселей в гистограмме: {np.sum(hist)}')
print(f'Площадь изображения = {h*w}')
'''распределение яркостей для равномерного распределения эквализацией'''
cdf = np.zeros(256, dtype = int)
cdf[0] = hist[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist[i]
'''LUT реализация формулы'''
lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    lut_v = 255 * (cdf[i] / cdf[255])
    lut[i] = np.round(lut_v).astype(np.uint8)
img_eq_gray = np.zeros_like(img_gray)
for y in range(h):
    for x in range(w):
        org_yark = img_gray[y, x]
        new_yark = lut[org_yark]
        img_eq_gray[y,x] = new_yark
'''конвертируем в hsv для изменения только яркости'''
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for y in range(h):
    for x in range(w):
        orig_v = img_hsv[y,x,2]
        img_hsv[y,x,2] = lut[orig_v]
        
img_eq_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
img_eq_color_rgb = cv2.cvtColor(img_eq_color, cv2.COLOR_BGR2RGB)
hist_eq = np.zeros(256, dtype = int)
for y in range(h):
    for x in range(w):
        yark = img_eq_gray[y,x]
        hist_eq[yark] += 1
plt.figure(figsize=(15, 10))

'''Визуализация с GridSpec 2x3'''
fig = plt.figure(figsize=(15, 8)) 
gs = plt.GridSpec(2, 3, figure=fig) 

'''Исходное изображение (ряд 0, колонка 0)'''
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img_rgb)
ax1.set_title('Исходное цветное изображение')
ax1.axis('off')

'''Эквализированное изображение '''
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img_eq_color_rgb)
ax2.set_title('После эквализации яркости')
ax2.axis('off')

'''LUT преобразование (ряд 0, колонка 2)'''
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(range(256), lut, 'green', linewidth=2)
ax3.set_title('Look-Up Table преобразования\nLUT[i] = 255 × CDF[i] / CDF[255]')
ax3.set_xlabel('Входная яркость')
ax3.set_ylabel('Выходная яркость')
ax3.grid(True)
ax3.set_xlim([0, 255])
ax3.set_ylim([0, 255])

'''Исходная гистограмма (ряд 1, колонка 0)'''
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(range(256), hist, color='blue', alpha=0.7, width=1.0)
ax4.set_title('Гистограмма исходной яркости')
ax4.set_xlabel('Яркость')
ax4.set_ylabel('Количество пикселей')
ax4.set_xlim([0, 255])

'''Гистограмма после эквализации (ряд 1, колонка 1)'''
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(range(256), hist_eq, color='red', alpha=0.7, width=1.0)
ax5.set_title('Гистограмма после эквализации')
ax5.set_xlabel('Яркость')
ax5.set_ylabel('Количество пикселей')
ax5.set_xlim([0, 255])

'''Сравнение гистограмм (ряд 1, колонка 2)'''
ax6 = fig.add_subplot(gs[1, 2])
ax6.bar(range(256), hist, color='blue', alpha=0.5, width=1.0, label='До')
ax6.bar(range(256), hist_eq, color='red', alpha=0.5, width=1.0, label='После')
ax6.set_title('Сравнение гистограмм')
ax6.set_xlabel('Яркость')
ax6.set_ylabel('Количество пикселей')
ax6.set_xlim([0, 255])
ax6.legend()

plt.tight_layout()
plt.show()