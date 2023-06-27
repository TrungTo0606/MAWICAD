#V5
import cv2
import numpy as np

def enhance_sharpness(image):
    # Tạo kernel cho sharpening filter
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Áp dụng sharpening filter
    imgEnhanced = cv2.filter2D(image, -1, kernel)

    return imgEnhanced

def enhanceHistV(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Chia thành các kênh màu
    h, s, v = cv2.split(hsv)

    # Cân bằng histogram trên kênh màu V
    equalized_v = cv2.equalizeHist(v)

    # Kết hợp các kênh màu lại
    enhanced_hsv = cv2.merge((h, s, equalized_v))
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return enhanced_image

def detectPlate(img):
    imgRaw = img
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for i in range(6):
        grayImg = cv2.medianBlur(grayImg, 5)

    imgBin = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgBin = cv2.morphologyEx(imgBin, cv2.MORPH_CLOSE, kernel=kernel_1, iterations=2)
    imgBin = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernel=kernel_1, iterations=1)

    # cắt bớt phần trên và dưới của tấm ảnh để tập trung vào giữa nơi có tấm phôi
    imgY = imgBin.shape[0]
    cutPercent = 16
    cut = (imgY*cutPercent) // 100    # đơn vị là phần trăm
    imgBinRoi = imgBin[cut:imgY-cut, :]

    contours, _ = cv2.findContours (imgBinRoi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    maxPlate = None
    for index, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        plateArea = cv2.contourArea(cnt)

        # bỏ đi các nhiễu nhỏ 
        if plateArea > 20000:
            perimeter = cv2.arcLength(cnt, True)
            err = abs(((w+h)*2) - perimeter)
            # tính giá trị chu vi để xác định xem có thực sự là hình chưx nhật không
            if err < 20000:
                # tìm giá trị contour lớn nhất
                if plateArea > maxArea:
                    maxArea = plateArea
                    y = y + cut
                    naiCutX = (w*5)//100    #đơn vị là phần trăm cắt bỏ phần dư của đinh trên trục X
                    nailCutY = (h*4)//100   # đơn vị là % cắt bỏ phần dư của đinh trên trục Y
                    maxPlate = imgRaw[y+nailCutY:y+h, x:x+w-naiCutX]
                    cv2.rectangle(imgRaw, (x, y+nailCutY), (x+w-naiCutX, y+h), (0, 255, 0), 2)

    if maxPlate is not None:
        return maxPlate, imgRaw
    else:
        print("Cannot detect plate")
        return None, imgRaw

def detectWelds(plate, img, numOfPlates):
    welds = []
    if plate is None:
        while len(welds) < 5:
            welds.append(None)
        return welds, img

    center = plate.shape[1]//(numOfPlates)
    weld_roi = (plate.shape[1] * 10) //100
    for i in range(1, numOfPlates):
        # weldRoi là vùng có thể chứa mối hàn
        weldRoi = plate[:, center*i - (weld_roi//2) : center*i + (weld_roi//2)]

        weldRoi = enhanceHistV(weldRoi)
        weldRoiHSV = cv2.cvtColor(weldRoi, cv2.COLOR_BGR2HSV)
        weldRoiBin = cv2.inRange(weldRoiHSV, (20, 5, 0), (150, 255, 110))

        # preprocessing for the kernel
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        weldRoiBin = cv2.morphologyEx(weldRoiBin, cv2.MORPH_OPEN, kernel=kernel_1, iterations=1)
        weldRoiBin = cv2.morphologyEx(weldRoiBin, cv2.MORPH_CLOSE, kernel=kernel_1, iterations=1)
        weldRoiBin = cv2.dilate(weldRoiBin, kernel_1, iterations=4)

        meanX1 = np.mean(weldRoiBin, axis=0)
        if np.max(meanX1) > 60:
            thresh = np.max(meanX1) - 40
            minX = np.min(np.where(meanX1 > thresh))
            maxX = np.max(np.where(meanX1 > thresh))

            # tính tâm mối hàn trong weldRoi
            weldCenter = (minX+maxX) // 2
            # tính sai số khoảng cách giữa tâm vùng quan tâm và tâm mối hàn trong vùng quan tâm cắt ra
            error = weldCenter - (weld_roi//2)
            # tính tâm mối hàn trong tấm phôi
            weldCenter = (center*i) + error

            weld_region = (plate.shape[1] * 12) // 100
            weld = plate[:, weldCenter - (weld_region//2):weldCenter + (weld_region//2)]
            weld = cv2.rotate(weld, cv2.ROTATE_90_CLOCKWISE)
            weld = cv2.resize(weld, [600, 160])

            welds.append(weld)
        else: 
            welds.append(None)

    while len(welds) < 5:
        welds.append(None)
    
    return welds, img

def imgProcessing(img, numOfPlates):
    plate, img_1 = detectPlate(img)
    welds, img_2 = detectWelds(plate, img, numOfPlates)

    return img_1, welds
