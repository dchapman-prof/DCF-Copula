import cv2
import numpy as np

layers = ['X', 'A', 'B', 'C', 'D']
FA     = [ 15,  14,  15,  72, 206]
FB     = [ 19,  40,  85, 245, 484]
series = ['leg', 'fou', 'his']

imgs = [[]*5]*3

bigimg = None

for j in range(3):
	ser = series[j]
	for i in range(5):
		lay = layers[i]
		a   = FA[i]
		b   = FB[i]
		
		#path = 'pitnd_paper\imagenette2_resnet18_0_%s_f_4_m_11_b_10/%s_%d_%d.png' % (lay, ser, a, b)
		path = 'pit_test/imagenette2_resnet18_0_%s_f_4_m_11_b_10/%s_%d_%d.png' % (lay, ser, a, b)
		
		print(path)
		img = cv2.imread(path)
		
		print(img.shape)
		
		if bigimg is None:
			sY,sX,ch = img.shape
			bigimg = np.zeros( (5*sY, 3*sX, ch), dtype=img.dtype)
			
		bigimg[(i*sY):((i+1)*sY), (j*sX):((j+1)*sX)] = img
		
cv2.imwrite('make_fig.png', bigimg)
