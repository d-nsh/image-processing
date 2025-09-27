# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 20:38:01 2025

@author: Даниш
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


'''Загружаем изображение'''
img = cv2.imread('orig.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

print(f'Размер изображения = {img_rgb.shape}')

'''ЛИНЕЙНАЯ ФИЛЬТРАЦИЯ В ПРОСТРАНСТВЕННОЙ ОБЛАСТИ'''
'''#Сглаживающие фильтры'''
kernel_blur = np.ones((5,5), np.float32) / 25  # Усредняющий фильтр
img_blur = cv2.filter2D(img_gray, -1, kernel_blur)

'''Фильтр Гаусса'''
kernel_gaussian = cv2.getGaussianKernel(5, 1.0)
kernel_gaussian = kernel_gaussian * kernel_gaussian.T
img_gaussian = cv2.filter2D(img_gray, -1, kernel_gaussian)

'''НЕЛИНЕЙНАЯ ФИЛЬТРАЦИЯ ПОЛУТОНОВЫХ'''
'''Медианный фильтр'''
img_median = cv2.medianBlur(img_gray.astype(np.uint8), 3).astype(np.float32)

'''Фильтр минимума/максимума (нелинейные)'''
def min_filter(image, size=3):
    return cv2.erode(image, np.ones((size,size)))

def max_filter(image, size=3):
    return cv2.dilate(image, np.ones((size,size)))

img_min = min_filter(img_gray.astype(np.uint8)).astype(np.float32)
img_max = max_filter(img_gray.astype(np.uint8)).astype(np.float32)

'''ПОВЫШЕНИЕ РЕЗКОСТИ'''

'''Фильтр Лапласа (высокочастотный)'''
kernel_laplace = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], np.float32)

img_laplace = cv2.filter2D(img_gray, -1, kernel_laplace)
img_sharp_laplace = img_gray + 0.5 * img_laplace

'''# Фильтр повышения резкости'''
kernel_sharpen = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]], np.float32) / 1.0

img_sharpened = cv2.filter2D(img_gray, -1, kernel_sharpen)

'''ПРЕОБРАЗОВАНИЕ ФУРЬЕ И ЧАСТОТНАЯ ФИЛЬТРАЦИЯ'''


'''Преобразование Фурье'''
dft = np.fft.fft2(img_gray)
dft_shift = np.fft.fftshift(dft)

'''Низкочастотный фильтр (сглаживание)'''
rows, cols = img_gray.shape
crow, ccol = rows//2, cols//2
mask_low = np.zeros((rows, cols), np.uint8)
mask_low[crow-30: crow+30, ccol-30: ccol+30] = 1  

'''Высокочастотный фильтр (резкость)'''
mask_high = 1 - mask_low  
'''Применяем фильтры'''
dft_low = dft_shift * mask_low
dft_high = dft_shift * mask_high

'''Обратное преобразование'''
img_fourier_low = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_low)))
img_fourier_high = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_high)))

'''ЭКВАЛИЗАЦИЯ ГИСТОГРАММЫ'''
img_eq = cv2.equalizeHist(img_gray.astype(np.uint8)).astype(np.float32)

'''КОМБИНИРОВАННЫЕ МЕТОДЫ'''

'''Сглаживание + эквализация'''
img_smooth_eq = cv2.equalizeHist(cv2.GaussianBlur(img_gray.astype(np.uint8), (3,3), 0))

'''Частотная фильтрация + эквализация'''
img_freq_eq = cv2.equalizeHist(img_fourier_low.astype(np.uint8))

'''ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ'''
plt.figure(figsize=(20, 15))

methods = [
    (img_gray, 'Исходное изображение', 'gray'),
    (img_blur, 'Линейная фильтрация\n(Усредняющий фильтр)', 'gray'),
    (img_gaussian, 'Линейная фильтрация\n(Фильтр Гаусса)', 'gray'),
    (img_median, 'Нелинейная фильтрация\n(Медианный фильтр)', 'gray'),
    (img_sharpened, 'Повышение резкости\n(Фильтр резкости)', 'gray'),
    (img_fourier_low, 'Частотная фильтрация\n(Низкочастотный)', 'gray'),
    (img_fourier_high, 'Частотная фильтрация\n(Высокочастотный)', 'gray'),
    (img_eq, 'Эквализация гистограммы', 'gray'),
    (img_smooth_eq, 'Комбинированный\n(Сглаживание + Эквализация)', 'gray')
]

for i, (image, title, cmap) in enumerate(methods):
    plt.subplot(3, 3, i+1)
    plt.imshow(image, cmap=cmap)
    plt.title(title, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.suptitle('СРАВНЕНИЕ МЕТОДОВ ОБРАБОТКИ ИЗОБРАЖЕНИЙ', fontsize=16, fontweight='bold')
plt.show()


