# USAGE
# python ocr_template_match.py --image images/credit_card_01.png --reference ocr_a_reference.png

# import the necessary packages
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", required=True,
	help="path to reference OCR-A image")
args = vars(ap.parse_args())

# define a dictionary that maps the first digit of a credit card
# number to the credit card type
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi

# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])


image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("g", gray)


kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
im = cv2.filter2D(gray, -1, kernel)

#New Layer
mid = cv2.GaussianBlur(im,(0,0),21,21)
th2 = cv2.adaptiveThreshold(mid,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
im2 = cv2.addWeighted(im,1.5,th2,-0.5,0)


cv2.imshow("im", im)
cv2.imshow("im2", im2)


# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the credit card numbers)

tophat = cv2.morphologyEx(im2, cv2.MORPH_TOPHAT, rectKernel)

#tophat = cv2.resize(tophat,(0,0),fx = 3,fy=3);
#tophat = cv2.resize(tophat,(0,0),fx = 6,fy=6)
# cv2.imshow("tophat", tophat)
# cv2.imshow("tophat2", tophat2)

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_16S, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")



# apply a closing operation using the rectangular kernel to help
# cloes gaps in between credit card number digits, then apply
# Otsu's thresholding method to binarize the image
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

#thresh = cv2.resize(thresh,(0,0),fx = 3,fy=3);
# cv2.imshow("thresh", thresh)
# find contours in the thresholded image, then initialize the
# list of digit locations
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []

# loop over the contours
largest_y = -1;
for (i, c) in enumerate(cnts):
	# compute the bounding box of the contour, then use the
	# bounding box coordinates to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	print("Getting the name")
	if y > largest_y:
		(x_name, y_name, w_name, h_name) = cv2.boundingRect(c)
		ar = w_name / float(h_name)
		#print (ar)
		print("=============")
		print (w)
		print (h)
		print ("===========")
		if ar > 2.5 and ar < 9.0: #consider chaning (min is 2.5, max is 4)
			if (w > 38 and w < 120) and (h > 10 and h < 20): 
				#cv2.rectangle(image, (x_name,y_name), (x_name+w_name, y_name+h_name),(0, 0, 255), 2)
				largest_y = y
				#cv2.imshow("ii", image)
	#cv2.waitKey(1000)
	if i > 0:
		(x2, y2, w2, h2) = cv2.boundingRect(cnts[i-1])
		if abs(y - y2) <= 1: 
			numbers_y_loc = y



for (i, c) in enumerate(cnts):
	# compute the bounding box of the contour, then use the
	# bounding box coordinates to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	if i > 0:
		(x2, y2, w2, h2) = cv2.boundingRect(cnts[i-1])
		if abs(y - y2) <= 1: 
			numbers_y_loc = y
			print ("numbers_y_loc")

for (i, c) in enumerate(cnts):
	# compute the bounding box of the contour, then use the
	# bounding box coordinates to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	#cv2.rectangle(image, (x,y), ((x+w), (y+h)), (0,0,0), 2) 
	# since credit cards used a fixed size fonts with 4 groups
	# of 4 digits, we can prune potential contours based on the
	# aspect ratio
	#cv2.rectangle(image, (x,y), (x+w, y+h),(0, 255, 0), 2)
	print ("===============================")
	cv2.imshow("aspectRatioOut", image)
	print (x)
	print (y)
	#cv2.waitKey(1500)
	if y >= numbers_y_loc:
		if ar > 2.5 and ar < 9.0: #consider chaning ()
			# contours can further be pruned on minimum/maximum width
			# and height
			#print ("height: ", h)
			#print ("width: ", w)
			if (w > 38 and w < 60) and (h > 10 and h < 20): #it was 38
				# append the bounding box region of the digits group
				# to our locations list
				cv2.rectangle(image, (x,y), (x+w, y+h),(0, 255, 0), 2)
				locs.append((x, y, w, h))
		else:
			locs.append((x_name, y_name, w_name, h_name))
			cv2.rectangle(image, (x_name,y_name), (x_name+w_name, y_name+h_name),(0, 255, 0), 2)

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locs = sorted(locs, key=lambda x:x[0])
output = []

# loop over the 4 groupings of 4 digits
for (i, (gX, gY, gW, gH)) in enumerate(locs):
	# initialize the list of group digits
	groupOutput = []
	cv2.rectangle(image, (gX,gY), ((gX+gW), (gY+gH)), (0,0,0), 2) 
	# extract the group ROI of 4 digits from the grayscale image,
	# then apply thresholding to segment the digits from the
	# background of the credit card
	group = im2[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	# #New Layer 
	# mid2 = cv2.GaussianBlur(group,(0,0),21,21)
	# th2 = cv2.adaptiveThreshold(mid2,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
 #            cv2.THRESH_BINARY,11,2)
	# g2 = cv2.addWeighted(group,1.5,th2,-0.5,0)
	cv2.imshow("group", group)
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	#cv2.waitKey(2000)
	# detect the contours of each individual digit in the group,
	# then sort the digit contours from left to right
	digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = imutils.grab_contours(digitCnts)
	digitCnts = contours.sort_contours(digitCnts,
		method="left-to-right")[0]

	# loop over the digit contours
	for c in digitCnts:
		# compute the bounding box of the individual digit, extract
		# the digit, and resize it to have the same fixed size as
		# the reference OCR-A images
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		#cv2.waitKey(1500)
		# initialize a list of template matching scores
		scores = []
		if ar  > 0.52 and ar < 0.8:
			
			#print ("AR: " , ar)
			#cv2.waitKey(1500)
			roi = group[y:y + h, x:x + w]
			roi = cv2.resize(roi, (57, 88))
			# loop over the reference digit name and digit ROI
			for (digit, digitROI) in digits.items():
				# apply correlation-based template matching, take the
				# score, and update the scores list
				result = cv2.matchTemplate(roi, digitROI,
					cv2.TM_CCOEFF)
				(_, score, _, _) = cv2.minMaxLoc(result)
				scores.append(score)

			# the classification for the digit ROI will be the reference
			# digit name with the *largest* template matching score
			if np.argmax(scores) == 10:
				groupOutput.append('/')
			else:	
				groupOutput.append(str(np.argmax(scores)))

			# draw the digit classifications around the group
			cv2.rectangle(image, (gX - 5, gY - 5),
			(gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
			cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

			# update the output digits list
	output.extend(groupOutput)
		#print ("==================")
		#print(np.argmax(scores))
		#print(scores)

# display the output credit card information to the screen
#print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
image = cv2.resize(image,(0,0),fx = 6,fy=6)
cv2.imwrite("t.png", group)
cv2.imshow("Image", image)
cv2.imwrite('final.png',image)
cv2.waitKey(0)