# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 22:31:55 2025

@author: Даниш
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
orig = cv2.imread('orig.jpg')
face = cv2.imread('face.jpg')
orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
h, w = face_gray.shape
res = cv2.matchTemplate(orig_gray, face_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
locations = np.where(res >= threshold)
new = orig.copy()
for pt in zip(*locations[::-1]):
    cv2.rectangle(new, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 7)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.title('Большая картинка')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
plt.title('Шаблон для поиска')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
plt.title(f'Найдено совпадений: {len(locations[0])}')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Найдено {len(locations[0])} совпадений с порогом {threshold}")