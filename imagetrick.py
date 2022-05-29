import tkinter # Biblioteka interfejsu graficznego
from tkinter import filedialog, Image, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw, ImageFont # pip install Pillow # Umożliwia otwieranie zdjęć, wstawiania do GUI oraz rysowanie zdjęć
import math
import cv2 as cv # pip install opencv-python
import numpy

def PillowToCV(pillowImage):
    """Konwertuj zdjęcie przystosowane do interfejsu graficznego na postać obrazu specjalnie przystosowanego do OpenCV"""
    array = numpy.asarray(pillowImage, order="F")
    if (pillowImage.mode == "RGB" or pillowImage.mode == "RGBA"):
        red = array[:,:,0].copy()
        blue = array[:,:,2].copy()
        array[:,:,0] = blue
        array[:,:,2] = red
        del red, blue
    return array

    #return numpy.asarray(pillowImage)

def CVToPillow(cvImage):
    """Konwertuj zdjęcie przystosowane do OpenCV na postać obrazu specjalnie przystosowanego do interfejsu graficznego"""
    img = cv.cvtColor(cvImage, cv.COLOR_BGR2RGB)
    return Image.fromarray(img)

class Graph():
    def __init__(self):
        self.width = 384
        self.height = 256
        self.data = []
    def generateGraph(self, width=384, height=256, data=[], maxValue=2, title="Graph", lineColor=(0, 0, 0)):
        if width < 384:
            self.width = 384
        elif width > 1976:
            self.width = 1976
        else:
            self.width = width
        if height < 256:
            self.height = 256
        elif height > 1536:
            self.height = 1536
        else:
            self.height = height

        self.lineColor = lineColor
        self.maxValue = int(maxValue)
        self.format = format
        self.title = title
        
        self.data = data
        self.graph = Image.new("RGB", (self.width, self.height), (255, 255, 255))

        self.font = ImageFont.truetype("arial.ttf", 48)

        self.graphDraw = ImageDraw.Draw(self.graph)
        
        self.lengthX = self.width - 129
        self.lengthY = self.height - 144
        self.size = self.graphDraw.textsize(self.title, self.font)
        self.graphDraw.text(((self.width/2)-self.size[0]/2, 16) , self.title, (0, 0, 0, 255), self.font)

        self.font = ImageFont.truetype("arial.ttf", 16)

        for num in range(3):
            self.graphNum = str(int(num * (256/2)))
            self.size = self.graphDraw.textsize(text=self.graphNum, font=self.font) # Zmienna pomocnicza

            self.graphDraw.text((64 + num/2 * (self.width - 2 * 64) - (self.size[0]/2), self.height - 32), self.graphNum, (0, 0, 0, 255), self.font)
            
            #self.graphDraw.line([(64 + num/2 * (self.width - 2 * 64), self.height - 56), (64 + num/2 * (self.width - 2 * 64), self.height - 40)], (16, 16, 16, 192), 3)
            self.graphDraw.line([(64 + num/2 * (self.width - 2 * 64), self.height - 48), (64 + num/2 * (self.width - 2 * 64), self.height - 40)], (16, 16, 16, 192), 3)
        
        self.graphDraw.line([(64, self.height - 48), (self.width - 64, self.height - 48)], (0, 0, 0, 255), 2) # Pozioma linia

        self.graphDraw.line([(64, self.height - 48), (64, 96)], (0, 0, 0, 255), 2) #Pionowa linia

        if (self.maxValue > 0):
            self.tmp = self.maxValue
            self.divide = 0
            while self.tmp > 10.0:
                self.divide = self.divide + 1
                self.tmp = self.tmp / 10
            self.tmp =  math.floor(int(self.tmp)) * 10**self.divide

            # Wypisz na pionowej linii wartości
            for vert in range(3):
                self.vertTmp = vert/2 * self.tmp
                if ((self.vertTmp) == round(self.vertTmp)):
                    self.vertTmp = int(self.vertTmp)
                self.size = self.graphDraw.textsize(text=str(self.vertTmp), font=self.font) # Zmienna pomocnicza
                self.graphDraw.text((62 - self.size[0], (self.height - 48 - self.size[1]/2) - ((self.vertTmp/self.maxValue) * self.lengthY)), 
                                    str(self.vertTmp), (0, 0, 0, 255), self.font) # Liczby na pionowej linii
                self.graphDraw.line( [(65, (self.height - 48) - ((self.vertTmp/self.maxValue) * self.lengthY)) ,
                                     (self.width - 64, (self.height - 48) - ((self.vertTmp/self.maxValue) * self.lengthY))], 
                                     (128, 128, 128, 128), 
                                     1 )
            
            # Wartości
            for val in range(len(self.data)):
                self.xVal = 0
                self.graphDraw.line( [(65 + round((val/len(self.data)) * self.lengthX) , self.height - 49),
                                      (65 + round((val/len(self.data)) * self.lengthX) , self.height - 49 - (self.lengthY * self.data[val]/self.maxValue))],
                                      self.lineColor, 1 )

        return self.graph
# ===================================
#  Program do egzaminu
# ===================================
class BackProjection_Mask():
    def __init__(self, pillowImage):
        self.low = 20
        self.up = 20
        self.pillowImage = numpy.array(pillowImage)
        self.pillowImage = cv.cvtColor(self.pillowImage, cv.COLOR_BGR2RGB)
        self.hsv = cv.cvtColor(self.pillowImage, cv.COLOR_BGR2HSV)

    def callback_low(self, val):
        self.low = val

    def callback_up(self, val):
        self.up = val


    def selectImagePosition(self):
        window_image = 'Wybierz punkt'
        cv.namedWindow(window_image)
        cv.imshow(window_image, self.pillowImage)

        cv.setMouseCallback(window_image, self.pickPoint)
        print('End')

    def pickPoint(self, event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN:
            return
        # Zdobądź współrzędne i w wybranym miejscu wypełnij maskowanie za pomocą klasycznego
        # wypełniania koloru
        seed = (x, y)
        newMaskVal = 255
        newVal = (120, 120, 120)
        connectivity = 8
        flags = connectivity + (newMaskVal << 8 ) + cv.FLOODFILL_FIXED_RANGE + cv.FLOODFILL_MASK_ONLY

        mask2 = numpy.zeros((self.pillowImage.shape[0] + 2, self.pillowImage.shape[1] + 2), dtype=numpy.uint8)
        cv.floodFill(self.pillowImage, mask2, seed, newVal, (self.low, self.low, self.low), (self.up, self.up, self.up), flags)
        mask = mask2[1:-1,1:-1]

        backproj = self.Hist_and_Backproj(mask)

        # Zwróc zdjęcie maski oraz projekcji wstecznej
        return (mask, backproj)

    def Hist_and_Backproj(self, mask):
        h_bins = 30
        s_bins = 32
        histSize = [h_bins, s_bins]
        h_range = [0, 180]
        s_range = [0, 256]
        ranges = h_range + s_range
        channels = [0, 1]

        # Wygeneruj histogram i znormalizuj go
        hist = cv.calcHist([self.hsv], channels, mask, histSize, ranges, accumulate=False)
        cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

        # Zapisz efekt pracy funkcji projekcji wstecznej
        backproj = cv.calcBackProject([self.hsv], channels, hist, ranges, scale=1)

        return backproj

#     def do(self):
#         # Read the image

#         #self.src = cv.imread('07b29190.jpeg')
#         self.src = self.pillowImage
#         if self.src is None:
#             print('Could not open or find the image:', self.pillowImage)
#             exit(0)

#         # Transform it to HSV
#         self.hsv = cv.cvtColor(self.src, cv.COLOR_BGR2HSV)

#         # Show the image
#         window_image = 'Source image'
#         cv.namedWindow(window_image)
#         cv.imshow(window_image, self.src)

#         # Set Trackbars for floodfill thresholds
#         cv.createTrackbar('Low thresh', window_image, self.low, 255, self.callback_low)
#         cv.createTrackbar('High thresh', window_image, self.up, 255, self.callback_up)
#         # Set a Mouse Callback
#         cv.setMouseCallback(window_image, pickPoint)

#         cv.waitKey()
# # ===================================
#  Koniec treści do egzaminu
# ===================================
class Program:
    def __init__(self):
        self.filename = ""
        self.image = None
        self.secondImage = None
        self.imageCV = None
        self.dataImage = []

        self.graphWidth = 384
        self.graphHeight = 256

        self.resultHistogram1 = None
        self.resultHistogram2 = None
        self.resultHistogram3 = None

    def exportCSV(self):
        if self.value is None: return

        self.s = filedialog.asksaveasfile(title="Eksportuj CSV do...", defaultextension=".csv", initialfile = '.'.join(self.filename.split('/')[-1].split('.')[:-1]),
        filetypes = (("Plik CSV","*.csv"), ("Wszystkie pliki","*.*")))

        if self.s is None: return None

        self.textCSV = ""

        if hasattr(self.value[0], "__getitem__"):
            for i in range(len(self.value)):
                self.first = None
                for j in self.value[i]:
                    if self.first is None:
                        self.first = True
                    else:
                        self.textCSV += ";"
                    self.textCSV += str(j)
                self.textCSV += "\n"
        else:
            self.first = None
            for j in self.value:
                if self.first is None:
                    self.first = True
                else:
                    self.textCSV += ";"
                self.textCSV += str(j)
            self.textCSV += "\n"

        self.s.write(self.textCSV)
        self.s.close()

    def openImage(self):
        """Otwiera zdjęcie"""
        self.filename = filedialog.askopenfilename(title = "Wybierz obraz", filetypes = (
            ("Obraz JPG","*.jpg;*.jpeg;.jpe;.jif;.jfif;.jfi"),
            ("Obraz PNG","*.png"),
            ("Obraz BMP","*.bmp"),
            ("Wszystkie pliki","*.*"))
            ) #Zapytaj, jakie zdjęcie ma być otwarte
        if self.filename != "": # Przypadek, gdy użytkownik wybierze zdjęcie
            try:
                if self.image is not None:
                    self.image.close()
                self.image = Image.open(self.filename)
                self.dataImage = list(self.image.getdata())
            except:
                messagebox.showerror("Błąd", "Nie zidentyfikowano zdjęcia!")
                return False
        else:
            return False
# ===================================
#  Program do egzaminu
# ===================================
    def Hist_and_Backproj(self, img, val):
        # Zamiana koloru "BGR" na postać HSV
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Dane pozwalające na wygenerowanie histogramu
        ch = (0, 0)
        hue = numpy.empty(hsv.shape, hsv.dtype)
        cv.mixChannels([hsv], [hue], ch)

        # Przypisanie wartości z suwaka (histBackScale) do funkcji
        bins = val
        histSize = max(bins, 2)
        ranges = [0, 180] # hue_range
        
        # Tworzenie histogramu oraz normalizacja wartości
        hist = cv.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
        cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        
        # Funkcja projekcji wstecznej
        backproj = cv.calcBackProject([hue], [0], hist, ranges, scale=1)
        
        
        #cv.imshow('BackProj', backproj)
        
        
        w = 500
        h = 500
        bin_w = int(round(w / histSize))
        histImg = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        for i in range(bins):
            cv.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(numpy.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv.FILLED)
        #cv.imshow('Histogram', histImg)

        return  (backproj, histImg)


    def backProjection(self,image=None):
        # original_image = cv.imread("stachu.jpg")
        # image = cv.imread("stachu2.jpg")

        # cv.imshow('original_image', original_image)
        # cv.imshow('image', image)

        def Hist_and_Backproj(val):
            # Inicjalizacja wartości
            bins = val
            histSize = max(bins, 2)
            ranges = [0, 180] # hue_range
            
            # Wygeneruj histogram i znormalizuj
            hist = cv.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
            cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            
            # Funkcja projekcji wstecznej
            backproj = cv.calcBackProject([hue], [0], hist, ranges, scale=1)
            
            
            cv.imshow('BackProj', backproj)
            
            # Narysuj histogram
            w = 400
            h = 400
            bin_w = int(round(w / histSize))
            histImg = numpy.zeros((h, w, 3), dtype=numpy.uint8)
            for i in range(bins):
                cv.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(numpy.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv.FILLED)
            cv.imshow('Histogram', histImg)
        
        # Odczytaj 
        src = cv.imread("Back_Projection_Theory2.jpg")
        if (image is not None):
            src = image
        if src is None:
            print('Could not open or find the image:', "stachu.jpg")
            exit(0)
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        ch = (0, 0)
        hue = numpy.empty(hsv.shape, hsv.dtype)
        cv.mixChannels([hsv], [hue], ch)
        window_image = 'Source image'
        cv.namedWindow(window_image)
        bins = 25
        cv.createTrackbar('* Hue  bins: ', window_image, bins, 180, Hist_and_Backproj )
        Hist_and_Backproj(bins)
        cv.imshow(window_image, src)
# ===================================
#  Koniec treści do egzaminu 
# ===================================
# ===================================
#  Histogram
# ===================================

    def histogram(self):
        """Tworzy histogram"""
        if self.validateImage() is None: return None

        if self.image.mode == "P": # Jeśli załadowane zdjęcie jest mapowalną paletą kolorów, to przerwij (brak wsparcia tworzenia histogramu dla takich)
            if messagebox.askyesno("Brak wsparcia", "Program wykrył, że zdjęcie posiada mapowalną paletę kolorów, przez który program nie jest w stanie stworzyć histogramu dla takiego obrazu. Ale jest możliwość, aby program przekonwertował to zdjęcie na paletę kolorów. Czy program ma przekształcić na paletę koloru i stworzyć na podstawie tego histogram dla koloru?"):
                self.image = self.image.convert(mode="RGB")
            else:
                return False
        elif hasattr(self.dataImage[0], "__getitem__") and (len(self.dataImage[0]) < 3):
            self.image = self.image.convert(mode="L")

        self.dataImage = list(self.image.getdata())

        if hasattr(self.dataImage[0], "__getitem__"): # Czy obraz jest kolorowy (czy są 3 kanały koloru)
            self.value = [256 * [0] for i in range(3)]
            self.maxValue = 0
            for channel in range(3):
                for val in self.dataImage:
                    self.value[channel][val[channel]] += 1
                if max(self.value[channel]) > self.maxValue:
                    self.maxValue = max(self.value[channel])

            self.results = [self.resultHistogram1, self.resultHistogram2, self.resultHistogram3]
            
            self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for channel in range(3):
                self.results[channel] = Graph().generateGraph(self.graphWidth, self.graphHeight, self.value[channel], self.maxValue, self.title[channel], self.colors[channel])
            self.maxValue = 0
            return self.results
                

        else:
            self.value = 256 * [0]

            for val in self.dataImage:
                self.value[val] += 1

            self.resultHistogram1 = Image.new("L", (256, 128), 255) # Stwórz obraz monochromatyczny dla histogramu
            
            self.maxValue = max(self.value)

            self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                
            # #Rysuj linię
            self.results = Graph().generateGraph(self.graphWidth, self.graphHeight, self.value, self.maxValue, self.title[3], (128, 128, 128))

            return [self.results, None, None]

        return None

    def stretchHistogram(self):
        """Rozciąganie histogramu"""
        if self.validateImage() is None: return None

        pictureSize = self.image.size

        values = self.value

        # Czy obraz jest kolorowy (czy są 3 kanały koloru)
        if hasattr(self.image.getdata()[0], "__getitem__"): 

            stretchTable = [[0.0 for i in range(256)] for i in range(3)]
            
            for i in range(3):
                min = 0
                max = 0
                # Min
                for j in range(0, 255, 1):
                    if values[i][j] != 0:
                        min = j
                        break

                for j in range(255, 0, -1):
                    if values[i][j] != 0:
                        max = j
                        break

                for j in range(len(values[i])):
                    stretchTable[i][j] = self.calcStretchHistogram(j, min, max)

            for y in range(pictureSize[1]):
                for x in range(pictureSize[0]):
                    r,g,b = self.image.getpixel((x, y))
                    self.image.putpixel((x, y), 
                    (stretchTable[0][r], stretchTable[1][g], stretchTable[2][b])) 

            del r,g,b
        else:
            stretchTable = 256 * [0.0]
            #Min
            for i in range(0, 255, 1):
                if values[i] != 0:
                    min = i
                    break
            # Max
            for i in range(255, 0, -1):
                if values[i] != 0:
                    max = i
                    break

            for i in range(len(stretchTable)):
                stretchTable[i] = self.calcStretchHistogram(i, min, max)

            for y in range(pictureSize[1]):
                for x in range(pictureSize[0]):
                    v = self.image.getpixel((x, y))
                    self.image.putpixel((x, y), 
                    stretchTable[v])

            del v, stretchTable
            
        del min, max, pictureSize, values
        #return None

    def histogramEqualization(self):
        if self.validateImage() is None: return None

        minValue = 0
        pixels = self.image.width * self.image.height

        # Czy obraz jest kolorowy (czy są 3 kanały koloru)
        if hasattr(self.image.getdata()[0], "__getitem__"): 
            length = len(self.value[0])
            distributor = [[0.0 for i in range(256)] for i in range(3)]

            for ch in range(3):
                sum = 0
                # Minimalna wartość
                for i in range(length):
                    if (self.value[ch][i] != 0.0):
                        minValue = self.value[ch][i]/pixels
                        break
                for i in range(length):
                    sum += self.value[ch][i]
                    distributor[ch][i] = int( ((sum/pixels) - minValue)/(1 - minValue) * 255 )

            for y in range(self.image.height):
                for x in range(self.image.width):
                    r,g,b = self.image.getpixel((x, y))
                    self.image.putpixel((x, y),
                    (distributor[0][r], distributor[1][g], distributor[2][b]))

            del r,g,b
        else:
            sum = 0

            length = len(self.value)
            distributor = length * [0.0]

            # Minimalna wartość
            for i in range(length):
                if (self.value[i] != 0.0):
                    minValue = self.value[i]/pixels
                    break
            for i in range(length):
                sum += self.value[i]
                distributor[i] = int( ((sum/pixels) - minValue)/(1 - minValue) * 255 )

            for y in range(self.image.height):
                for x in range(self.image.width):
                    v = self.image.getpixel((x, y))
                    self.image.putpixel((x, y),
                    distributor[v])

            del v

        del sum, minValue, distributor, length

    def createHistogram(self):
        """Stwórz histogram na podstawie otwartego zdjęcia"""
        return self.histogram()

    def calcStretchHistogram(self, val, min, max):
        # Wzór z wykładu nr 2
        return int((val - min)*(255/(max-min)))

    def negation(self):
        if hasattr(self.image.getdata()[0], "__getitem__"):
            for y in range(self.image.height):
                for x in range(self.image.width):
                    r,g,b = self.image.getpixel((x, y))
                    self.image.putpixel((x, y),
                    (255-r, 255- g, 255-b))
        else:
            for y in range(self.image.height):
                for x in range(self.image.width):
                    v = self.image.getpixel((x, y))
                    self.image.putpixel((x, y),
                    255-v)

    def thresholding(self, value=128, keepColors=False, image=None):
        outSource = True
        if (image is None):
            image = self.image
            outSource = False

        if hasattr(image.getdata()[0], "__getitem__"):
            for y in range(image.height):
                for x in range(image.width):
                    r,g,b = image.getpixel((x, y))
                    if (keepColors):
                        image.putpixel((x, y),
                        (0 if r < value else r,
                        0 if g < value else g,
                        0 if b < value else b))
                    else:
                        image.putpixel((x, y),
                        (0 if r < value else 255,
                        0 if g < value else 255,
                        0 if b < value else 255))
        else:
            for y in range(image.height):
                for x in range(image.width):
                    v = image.getpixel((x, y))
                    if (keepColors):
                        image.putpixel((x, y),
                        0 if v < value else v)
                    else:
                        image.putpixel((x, y),
                        0 if v < value else 255)

        return image

    def applyPosterization(self, value=2):
        if self.image == None: return None
        self.image = self.posterizationPreview(self.image, value)

    def posterizationPreview(self, image, value=2):

        image = image.copy()

        if(value < 2):
            value = 2
        value = value - 1
        divide = 256/value
        
        if hasattr(image.getdata()[0], "__getitem__"):
            for y in range(image.height):
                for x in range(image.width):
                    r,g,b = image.getpixel((x, y))
                    image.putpixel((x, y),
                    (round(round(r/divide) * divide), round(round(g/divide) * divide), round(round(b/divide) * divide)))
        else:
            for y in range(image.height):
                for x in range(image.width):
                    v = image.getpixel((x, y))
                    image.putpixel((x, y),
                    round(round(v/divide) * divide))

        return image

    def polerization(self, value=2):

        if hasattr(self.image.getdata()[0], "__getitem__"):
            for y in range(self.image.height):
                for x in range(self.image.width):
                    r,g,b = self.image.getpixel((x, y))
                    self.image.putpixel((x, y),
                    (255-r, 255- g, 255-b))
        else:
            for y in range(self.image.height):
                for x in range(self.image.width):
                    v = self.image.getpixel((x, y))
                    self.image.putpixel((x, y),
                    255-v)

# ===================================
#  Operacje wygładzania
# ===================================
    def gaussianBlur(self, border=cv.BORDER_DEFAULT):
        if self.image is None: return
        imgCv = PillowToCV(self.image)
        imgCv = cv.GaussianBlur(imgCv, (3,3), borderType=border)
        return CVToPillow(cv.GaussianBlur(imgCv, (3,3), borderType=border))

    def blur(self, border=cv.BORDER_DEFAULT):
        if self.image is None: return
        imgCv = PillowToCV(self.image)
        imgCv = cv.blur(imgCv, (3,3), borderType=border)
        return CVToPillow(cv.blur(imgCv, (3,3), borderType=border))
        

# ===================================
#  Operacje morfologiczne
# ===================================

    def erosion(self, area = 3, img=None, mode=None):
        if self.image is None and img is None: return
        kernel = numpy.ones((area, area), numpy.uint8)
        if(mode is not None): # Rhombus
            if(mode == 1):
                for i in range(area):
                    for j in range(area):
                        length = (area - 1) / 2
                        if ((abs(length - i) + abs(length - j)) > length):
                            kernel[i][j] = 0
        
        if(img is not None):
            imgCv = cv.erode(PillowToCV(img), kernel, iterations=1)
            del kernel
            return CVToPillow(imgCv)
        elif(self.image is not None):
            self.imageCV = cv.erode(PillowToCV(self.image), kernel, iterations=1)
            self.image = CVToPillow(self.imageCV)
            del kernel
            return self.image

    def dilation(self, area=3, img=None, mode=None):
        if self.image is None and img is None: return
        kernel = numpy.ones((area, area), numpy.uint8)
        if(mode is not None): # Rhombus
            if(mode == 1):
                for i in range(area):
                    for j in range(area):
                        length = (area - 1) / 2
                        if ((abs(length - i) + abs(length - j)) > length):
                            kernel[i][j] = 0
        
        if(img is not None):
            imgCv = cv.dilate(PillowToCV(img), kernel, iterations=1)
            del kernel
            return CVToPillow(imgCv)
        elif(self.image is not None):
            self.imageCV = cv.dilate(PillowToCV(self.image), kernel, iterations=1)
            self.image = CVToPillow(self.imageCV)
            del kernel
            return self.image

    def opening(self, area = 3, img=None, mode=None):
        if self.image is None and img is None: return
        kernel = numpy.ones((area, area), numpy.uint8)
        if(mode is not None): # Rhombus
            if(mode == 1):
                for i in range(area):
                    for j in range(area):
                        length = (area - 1) / 2
                        if ((abs(length - i) + abs(length - j)) > length):
                            kernel[i][j] = 0

        if(img is not None):
            imgCv = cv.morphologyEx(PillowToCV(img), cv.MORPH_OPEN, kernel, iterations=1)
            del kernel
            return CVToPillow(imgCv)
        elif(self.image is not None):
            self.imageCV = cv.morphologyEx(PillowToCV(self.image), cv.MORPH_OPEN, kernel, iterations=1)
            self.image = CVToPillow(self.imageCV)
            del kernel
            return self.image

    def closing(self, area = 3, img=None, mode=None):
        if self.image is None and img is None: return
        kernel = numpy.ones((area, area), numpy.uint8)
        if(mode is not None): # Rhombus
            if(mode == 1):
                for i in range(area):
                    for j in range(area):
                        length = (area - 1) / 2
                        if ((abs(length - i) + abs(length - j)) > length):
                            kernel[i][j] = 0

        if(img is not None):
            imgCv = cv.morphologyEx(PillowToCV(img), cv.MORPH_CLOSE, kernel, iterations=1)
            del kernel
            return CVToPillow(imgCv)
        elif(self.image is not None):
            self.imageCV = cv.morphologyEx(PillowToCV(self.image), cv.MORPH_CLOSE, kernel, iterations=1)
            self.image = CVToPillow(self.imageCV)
            del kernel
            return self.image

        if self.image is None: return
        
        imgCv = PillowToCV(self.image.convert("L"))

        size = numpy.size(imgCv)
        skel = numpy.zeros(imgCv.shape, numpy.uint8)
        ret,imgCv = cv.threshold(imgCv,127,255,0)
        element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

        while(True):
            eroded = cv.erode(imgCv,element)
            temp = cv.dilate(eroded,element)
            temp = cv.subtract(imgCv, temp)
            skel = cv.bitwise_or(skel, temp)
            imgCv = eroded.copy()

            zeros = size - cv.countNonZero(imgCv)
            if (zeros==size):
                break
        
        return CVToPillow(skel)

    def watershed(self):
        # Konwetuj każde zdjęcie z Pillowa na obraz kolorowy i z powrotem na odcień szarości.
        # W ten sposób unikam błędu przy ładowania zdjęcia szaro odcieniowego
        img = PillowToCV(self.image.convert("RGB"))
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # Progowanie czarno białe
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # Usuwanie najmniejszych obiektów, szczegółów zbędnych używając operacji morfologicznej otwieranie
        kernel = numpy.ones((3,3),numpy.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN,kernel, iterations = 1)

        # Obszar tła, używając dylacji
        sure_bg = cv.dilate(opening,kernel,iterations=1)

        # Obszary obiektów
        #  Transofrmacja odległościowa
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2,5)
        #  Określenie jednoznacznych obszarów obiektów przez progowanie obrazu transformaty odlegościowej
        ret, sure_fg = cv.threshold(dist_transform,0.5*dist_transform.max(),255,0)
        sure_fg = numpy.uint8(sure_fg)

        # Odejmowanie kolorów dla określenia obiektów nachodzących się między sobą
        unknown = cv.subtract(sure_bg,sure_fg)
        # Etykietowanie obiektów
        ret, markers = cv.connectedComponents(sure_fg)
        countObj = numpy.max(markers)

        # dodanie wartości 1 do etykiet, tak aby tło miało wartość 1 a nie 0.
        markers = markers+1
        # oznaczenie obszarów nachodzących się jako zero
        markers[unknown==255] = 0

        #Algorytm wododziału (watershed)
        markers = cv.watershed(img,markers)
        img[markers == -1] = [255,0,255]

        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return (img, countObj)

    def watershedSteps(self, img=None):
        steps = []

        # Konwetuj każde zdjęcie z Pillowa na obraz kolorowy i z powrotem na odcień szarości.
        # W ten sposób unikam błędu przy ładowania zdjęcia szaro odcieniowego
        if (img is not None):
            img = PillowToCV(img.convert("RGB"))
        else:
            img = PillowToCV(self.image.convert("RGB"))
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        steps.append(gray)

        # Progowanie czarno białe
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        steps.append(thresh)
        # Usuwanie najmniejszych obiektów, szczegółów zbędnych używając operacji morfologicznej otwieranie
        kernel = numpy.ones((3,3),numpy.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN,kernel, iterations = 1)
        steps.append(opening)

        # Obszar tła, używając dylacji
        sure_bg = cv.dilate(opening,kernel,iterations=1)
        steps.append(sure_bg)

        # Obszary obiektów
        #  Transofrmacja odległościowa
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2,5)
        steps.append(cv.cvtColor(cv.applyColorMap(numpy.uint8(dist_transform*10), cv.COLORMAP_JET), cv.COLOR_BGR2RGB))
        #  Określenie jednoznacznych obszarów obiektów przez progowanie obrazu transformaty odlegościowej
        ret, sure_fg = cv.threshold(dist_transform,0.5*dist_transform.max(),255,0)
        sure_fg = numpy.uint8(sure_fg)
        steps.append(sure_fg)

        # Odejmowanie kolorów dla określenia obiektów nachodzących się między sobą
        unknown = cv.subtract(sure_bg,sure_fg)
        steps.append(unknown)
        # Etykietowanie obiektów
        ret, markers = cv.connectedComponents(sure_fg)
        countObj = numpy.max(markers)
        steps.append(cv.cvtColor(cv.applyColorMap(numpy.uint8(markers*10), cv.COLORMAP_JET), cv.COLOR_BGR2RGB))

        # dodanie wartości 1 do etykiet, tak aby tło miało wartość 1 a nie 0.
        markers = markers+1
        # oznaczenie obszarów nachodzących się jako zero
        markers[unknown==255] = 0
        steps.append(cv.cvtColor(cv.applyColorMap(numpy.uint8(markers*10), cv.COLORMAP_JET), cv.COLOR_BGR2RGB))

        #Algorytm wododziału (watershed)
        markers = cv.watershed(img,markers)
        #steps.append(markers)
        img[markers == -1] = [255,0,255]
        steps.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        steps.append(countObj)
        return steps

# ===================================

    def getDataImage(self):
        """Zwróć dane pikseli"""
        return self.dataImage

    def getImage(self):
        """Zwróć zdjęcie"""
        return self.image

    def validateImage(self):
        if not self.image: return None # Jeśli nie ma zdjęcia załadowanego, to zakończ polecenie
        return True     


class Window:
    def __init__(self):
        self.okno = tkinter.Tk() #Inicjuj główne okienko
        self.okno.bind("<Configure>", self.oknoUpdate) # Po zmianie wielkości okna wykonaj tą komendę

        self.displayedImage = None
        self.program = Program()

        self.okno.title('Image Trick')
        self.okno.geometry("640x480")
        self.okno.minsize(640, 480)
        #self.okno.wm_resizable(width=False, height=False)

        #self.displayedImage = None
        self.histogram = None

        # Wartości domyślne
        self.autoRefresh = tkinter.BooleanVar()
        self.autoRefresh.set(True)
        self.widthGraphValue = 640
        self.heightGraphValue = 400
        self.bgColorGraphValue = '#fff'
        self.clrLineGraphValue = '#000'
        self.clrHorizontalGraphValue = '#808080'
        self.RedHistogramTitle = "Histogram - kanał czerwony"
        self.GreenHistogramTitle = "Histogram - kanał zielony"
        self.BlueHistogramTitle = "Histogram - kanał niebieski"
        self.GrayHistogramTitle = "Histogram - odcień szarości"

        self.smoothMode = tkinter.IntVar()
        self.smoothBorder = tkinter.IntVar()
        self.smoothBorder.set(cv.BORDER_REFLECT)
        self.morphologyMode = tkinter.IntVar()
        self.morphologyShape = tkinter.IntVar()
        self.morphologySize=3
        self.morphologySecondSize=3
        self.waterizetextInfo = tkinter.StringVar()

        #Stwórz pasek menu
        self.menu = tkinter.Menu(self.okno)

        # Pole w pasku menu #
        #-------------------
        #Plik
        self.file_menu = tkinter.Menu(self.menu, tearoff=0)
        self.file_menu.add_command(label="Otwórz", command=self.OpenImage)
        self.file_menu.add_command(label="Zapisz jako...", state="disabled", command=self.saveImage)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Zapisz histogram(y)", state="disabled", command=self.saveHistogram)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Wyjdź", command=self.okno.destroy)

        self.menu.add_cascade(label="Plik", menu=self.file_menu)
        #Akcja
        self.action_menu = tkinter.Menu(self.menu, tearoff=0)
        self.action_menu.add_command(label="Histogram", state="disabled", command=self.createHistogram)
        self.action_menu.add_command(label="Rozciąganie histogramu", state="disabled", command=self.doStretch)
        self.action_menu.add_command(label="Wyrównywanie histogramu", state="disabled", command=self.doEqualization)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Negacja", state="disabled", command=self.doNegation)
        self.action_menu.add_command(label="Progowanie", state="disabled",command=self.doThresholding)
        self.action_menu.add_command(label="Posteryzacja", state="disabled", command=self.doPosterization)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Wygładzanie", state="disabled", command=self.doSmoothing)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Operacje dwuargumentowe", state="disabled", command=self.doArgument)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Operacje morfologiczne", state="disabled", command=self.doMorphology)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Wododział", state="disabled", command=self.doWatersheding)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Projekcja wsteczna", state="disabled", command=self.doBackProjection)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Opcje", command=self.options)

        self.menu.add_cascade(label="Akcja", menu=self.action_menu)
        #-------------------
        self.okno.config(menu=self.menu) # Wstaw przygotowany pasek menu
        self.okno.grid()

        #self.okno_program = tkinter.Frame(self.okno)
        #------------------------------------------------
        # Ramka, w której będzie wyświetlone oryginalne zdjęcie
        self.photo_label = tkinter.LabelFrame(master=self.okno, width=300, height=150, text="Obraz")
        self.photo_show_label = None
        self.photo_label.grid(padx=5, row=0, column=0, columnspan=3, sticky='NSEW')

        self.histogram_label = tkinter.LabelFrame(master=self.okno, width=300, height= 150)
        self.histogram_show_label = None
        self.histogram_label.grid(padx=5, row=1, column=0, sticky='NSEW')

        self.histogram2_label = tkinter.LabelFrame(master=self.okno, width=300, height=150)
        self.histogram2_show_label = None
        self.histogram2_label.grid(padx=5, row=1, column=1, sticky='NSEW')
        
        self.histogram3_label = tkinter.LabelFrame(master=self.okno, width=300, height=150)
        self.histogram3_show_label = None
        self.histogram3_label.grid(padx=5, row=1, column=2, sticky='NSEW')

        # --------------------------------------------------------------

        self.tools_label = tkinter.Frame(master=self.okno, width=300)

        self.checkBoxAutoHistogram = tkinter.Checkbutton(self.tools_label, text="Odśwież histogram po zmianie", var = self.autoRefresh)
        self.checkBoxAutoHistogram.grid(row=0, column=0, sticky="SE")

        self.stretchScale = tkinter.Spinbox(self.tools_label, state="disabled", from_=0, to=255, width=4, command=self.rgbValue)
        self.stretchScale.grid(row=0,  column=1, sticky="E")
       

        self.textValue = tkinter.Label(self.tools_label, anchor="w", justify="right", state="disabled", text="Czerwony: %s; Zielony: %s; Niebieski: %s"%("", "", ""))
        self.textValue.grid(row=0, column=2, sticky="SE")

        self.buttonCsv = tkinter.Button(self.tools_label, state="disabled", text="Eksportuj do CSV", command=self.program.exportCSV)
        self.buttonCsv.grid(row=0, column=3, sticky="E",)

        self.tools_label.grid(padx=5, pady=5, column=0, columnspan=3, row=3,  sticky="NSEW")

        self.okno.grid_columnconfigure(0,weight=1)
        self.okno.grid_columnconfigure(1,weight=1)
        self.okno.grid_columnconfigure(2,weight=1)
        self.okno.grid_rowconfigure(0,weight=1)
        self.okno.grid_rowconfigure(1,weight=1)
     
        self.tools_label.grid_columnconfigure(3, weight=1)

        self.morphologyImagePreview = None

        # Działanie okienka
        self.okno.mainloop()

    def rgbValue(self):
        #value = int(value) - 1
        value = int(self.stretchScale.get()) #int(value) - 1
        if hasattr(self.program.value[0], "__getitem__"):
            self.textValue['text'] = "Czerwony: " + str(self.program.value[0][value]) +"; Zielony: " + str(self.program.value[1][value]) + "; Niebieski: " + str(self.program.value[2][value])
        else:
            self.textValue['text'] = "Wartość: " + str(self.program.value[value]) + " (odcień szarości)"

    def oknoUpdate(self, data=None):
        """Scaluj zdjęcia tak, aby się wpasowały dookoła wnętrza okna"""
        # Bardzo kosztowny proces, ponieważ przy scalaniu okna zostają generowane nowe zdjęcia, które będą dopasowane do pola wewnątrz. Nie mam pomysłu, jak złagodzić taki proces
        if self.photo_show_label is not None:
            self.scaleImage(self.program.image, self.photo_label, self.photo_show_label)
        if self.histogram_show_label is not None:
            self.scaleImage(self.histogram[0], self.histogram_label, self.histogram_show_label)
        if self.histogram2_show_label is not None:
            self.scaleImage(self.histogram[1], self.histogram2_label, self.histogram2_show_label)
        if self.histogram3_show_label is not None:
            self.scaleImage(self.histogram[2], self.histogram3_label, self.histogram3_show_label)

    def check(self):
        """Wywołanie komendy z przycisku "Check" """
        self.sampleGraph = ImageTk.PhotoImage(Graph().generateGraph(800, 500))
        if self.photo_show_label is None:
            self.photo_show_label = tkinter.Label(master=self.photo_label, width=320-24, height=200-24, image=self.sampleGraph)
            self.photo_show_label.pack(fill="both", expand=True)
        else:
            self.photo_show_label['image'] = self.sampleGraph
            self.photo_show_label.image = self.sampleGraph

    def scaleImage(self, image, frame, photoFrame):
        if (image is None): 
            photoFrame['image'] = None
            return None

        self.size = image.size
        self.frame = frame
        self.photoFrame = photoFrame
        self.imageRatio = self.size[0]/self.size[1]


        if (self.frame.winfo_width()/self.frame.winfo_height()) > self.imageRatio:
            self.newSize = (int(self.size[0] * (self.frame.winfo_height()/self.size[1])), self.frame.winfo_height())
        elif (self.frame.winfo_width()/self.frame.winfo_height()) == self.imageRatio:
            return None # Nie ma potrzeby scalać zdjęcia
        else:
            self.newSize = (int(self.frame.winfo_width()), int(self.size[1] * (self.frame.winfo_width()/self.size[0])))

        self.displayedImage = ImageTk.PhotoImage(image.resize((self.newSize), Image.LANCZOS))
        if self.photoFrame is None:
            self.photoFrame = tkinter.Label(master=self.frame, width=320-24, height=200-24, image=self.displayedImage)
            self.photoFrame.pack(fill="both", expand=True)
        else:
            self.photoFrame['image'] = self.displayedImage
            self.photoFrame.image = self.displayedImage


    def OpenImage(self):
        # Sprawdź, czy użytkownik wybrał zdjęcie
        if self.program.openImage() is False: return None

        self.enableOptions("normal")
        self.updateThumbnail(True, self.autoRefresh.get())
            
    def saveImage(self):
        imageFile = filedialog.asksaveasfilename(title="Zapisz obraz jako...", defaultextension=".png",
        filetypes = (
            ("Obraz JPG","*.jpg ;*.jpeg"),
            ("Obraz PNG","*.png"),
            ("Obraz BMP","*.bmp"),
            ))
        if (imageFile is None or imageFile == ''):
            return
        typeFile = imageFile.split('.')[-1]
        if(typeFile.upper() == "JPG"):
            typeFile = "JPEG"
        self.program.image.save(imageFile, format=typeFile)
        del imageFile, typeFile
        

    def saveHistogram(self):        
        if (self.histogram is None): 
            messagebox.showerror("Błąd", "Brak wygenerowanego histogramu!")
            return

        self.s = filedialog.asksaveasfilename(title="Zapisz histogram do...", defaultextension=".png", initialfile = '.'.join(self.program.filename.split('/')[-1].split('.')[:-1]),
        filetypes = (
            ("Obraz PNG","*.png"),
            ("Obraz BMP","*.bmp"),
            ))

        if (self.s == "" or self.s is None): return

        self.route = '/'.join(self.s.split('/')[:-1]) + '/'
        self.s = ['.'.join(self.s.split('/')[-1].split('.')[:-1]) , 'histogram' , self.s.split('/')[-1].split('.')[-1]]

        if (self.histogram[1] != None):
            for i in range(3): 
                self.s[1] = "histogram" + str(i+1)

                self.histogram[i].save(self.route + '.'.join(self.s), self.s[-1])
        else:
            self.histogram[0].save(self.route + '.'.join(self.s))
        self.route = None

    def enableOptions(self, state="normal"):
        """Umożliwia odblokowanie opcji menu"""
        # Must be active, normal, or disabled
        try:
            self.file_menu.entryconfigure(1, state=state)
            self.action_menu.entryconfigure(0, state=state) # Histogram
            self.action_menu.entryconfigure(1, state=state) # Rozciąganie histogramu
            self.action_menu.entryconfigure(2, state=state) # Wyrównywanie histogramu
            self.action_menu.entryconfigure(4, state=state) # Negacja
            self.action_menu.entryconfigure(5, state=state) # Progowanie
            self.action_menu.entryconfigure(6, state=state) # Posteryzacja
            self.action_menu.entryconfigure(8, state=state) # Wygładzanie
            self.action_menu.entryconfigure(10, state=state) # Operacje dwuargumentowe
            self.action_menu.entryconfigure(12, state=state) # Operacje morfologiczne
            self.action_menu.entryconfigure(14, state=state) # Wododział
            self.action_menu.entryconfigure(16, state=state) # Projekcja wsteczna
            return True
        except:
            return False

    def updateThumbnail(self, deleteThumbnail=False, updateHistogram = False):
        self.size = self.program.image.size
        
        self.imageRatio = self.size[0]/self.size[1]

        if (self.photo_label.winfo_width()/self.photo_label.winfo_height()) > self.imageRatio:
            self.newSize = (int(self.size[0] * (self.photo_label.winfo_height()/self.size[1])), self.photo_label.winfo_height())
        else:
            self.newSize = (int(self.photo_label.winfo_width()), int(self.size[1] * (self.photo_label.winfo_width()/self.size[0])))

        # Wygeneruj nowe zdjęcie do podglądu
        self.displayedImage = ImageTk.PhotoImage(self.program.image.resize((self.newSize), Image.LANCZOS))
        if self.photo_show_label is None:
            self.photo_show_label = tkinter.Label(master=self.photo_label, width=320-24, height=200-24, image=self.displayedImage)
            self.photo_show_label.pack(fill="both", expand=True)
        else:
            self.photo_show_label['image'] = self.displayedImage
            self.photo_show_label.image = self.displayedImage
            
        # Wyczyść dane histogramu
        if deleteThumbnail == True:
            self.histogram = None
            # Usuń poprzednio wygenerowane histogramy z innego zdjęcia
            self.stretchScale['state'] = "disabled"
            self.textValue['state'] = "disabled"
            self.buttonCsv['state'] = "disabled"
            self.file_menu.entryconfigure(3, state="disabled")
            self
            if self.histogram_show_label is not None:
                self.histogram_show_label.pack_forget()
                self.histogram_show_label = None
            if self.histogram2_show_label is not None:
                self.histogram2_show_label.pack_forget()
                self.histogram2_show_label = None
            if self.histogram3_show_label is not None:
                self.histogram3_show_label.pack_forget()
                self.histogram3_show_label = None

        if updateHistogram == True:
            self.createHistogram()

    def createHistogram(self):

        """Stwórz i pobierz z klasy Program histogram"""
        self.program.title = [self.RedHistogramTitle, self.GreenHistogramTitle, self.BlueHistogramTitle, self.GrayHistogramTitle]
        self.program.graphWidth = self.widthGraphValue
        self.program.graphHeight = self.heightGraphValue
        self.histogram = self.program.createHistogram()

        self.enableOptions('normal')
        
        # Gdy nie załadowano zdjęcia
        if self.histogram is None:
            messagebox.showerror("Błąd", "Nie załadowano żadnego zdjęcia!")
            return None
        

        self.displayHistogram = [ImageTk.PhotoImage(i) for i in self.histogram if i is not None]

        # Sprawdź, czy istnieje histogram dla danego kanału. Jeśli tak, to sprawdź, czy warstwa wyświetlająca obraz została stworzona
        if self.histogram[0] is not None:
            if self.histogram_show_label is None:
                self.histogram_show_label = tkinter.Label(master=self.histogram_label, width=291, height=178, image=self.displayHistogram[0])
                self.histogram_show_label.pack(fill="both", expand=True)
            else:
                self.histogram_show_label['image'] = self.displayHistogram[0]
                self.histogram_show_label.image = self.displayHistogram[0]
        if self.histogram[1] != None:
            if self.histogram2_show_label is None:
                self.histogram2_show_label = tkinter.Label(master=self.histogram2_label, width=291, height=178, image=self.displayHistogram[1])
                self.histogram2_show_label.pack(fill="both", expand=True)
            else:
                self.histogram2_show_label['image'] = self.displayHistogram[1]
                self.histogram2_show_label.image = self.displayHistogram[1]
        if self.histogram[2] != None:
            if self.histogram3_show_label is None:
                self.histogram3_show_label = tkinter.Label(master=self.histogram3_label, width=291, height=178, image=self.displayHistogram[2])
                self.histogram3_show_label.pack(fill="both", expand=True)
            else:
                self.histogram3_show_label['image'] = self.displayHistogram[2]
                self.histogram3_show_label.image = self.displayHistogram[2]

        # Aktywuj opcję do histogramu
        self.stretchScale['state'] = "normal"
        self.textValue['state'] = "active"
        self.buttonCsv['state'] = "active"
        self.file_menu.entryconfigure(3, state="normal")
        #self.rgbValue(self.stretchScale.get())
        self.rgbValue()
        self.oknoUpdate()

    def doStretch(self):
        # self.graphOption_label.destroy()
        if self.program.image == None: return None
        self.program.stretchHistogram()
        self.updateThumbnail(False, self.autoRefresh.get())
        self.oknoUpdate()
    
    def doEqualization(self):
        if self.program.image == None: return None
        self.program.histogramEqualization()
        self.updateThumbnail(False, self.autoRefresh.get())
        self.oknoUpdate()

    def doNegation(self):
        if self.program.image == None: return None
        self.program.negation()
        self.updateThumbnail(False, self.autoRefresh.get())
        self.oknoUpdate()

    def doThresholding(self):
        if self.program.image is None: return
        
        self.sizeThresholding = (300, 300)
        self.checkThresholding = tkinter.BooleanVar()
        self.checkThresholding.set(False)

        imageRatio = self.program.image.width/self.program.image.height

        if (1 > imageRatio):
            self.sizeThresholding = (int(self.program.image.width * (300/self.program.image.height)), 300)
        else:
            self.sizeThresholding = (300, int(self.program.image.height * (300/self.program.image.width)))

        self.ThresholdingValue = tkinter.IntVar()
        self.ThresholdingValue.set(128)

        self.previewEditedImage = self.program.thresholding(self.ThresholdingValue.get(), self.checkThresholding.get(), self.program.image.resize((self.sizeThresholding), Image.LANCZOS))
        previewImage = ImageTk.PhotoImage(self.previewEditedImage, Image.LANCZOS)

        # Okienko
        self.thresholdingMenu = tkinter.Toplevel()
        self.thresholdingMenu.title("Progowanie")
        self.thresholdingMenu.resizable(False, False)
        self.thresholdingMenu.grab_set()
        self.thresholdingMenu.focus()

        self.thresholding_label = tkinter.LabelFrame(master=self.thresholdingMenu, padx=10, pady=10, text="Progowanie")
        self.thresholding_label.grid(row=0, column=0)

        self.thresholdingPreview = tkinter.Label(master=self.thresholding_label, image=previewImage)
        self.thresholdingPreview.image = previewImage
        self.thresholdingPreview.grid(row=0, column=0, columnspan=3)

        self.thresholdingKeepCheck = tkinter.Checkbutton(self.thresholding_label, text="Zachowaj poziomy szarości / kolorów", var = self.checkThresholding)
        self.thresholdingKeepCheck.grid(row=1, column=0, columnspan=3)

        self.thresholdingScale = tkinter.Scale(master=self.thresholding_label, length=300, from_=0, to=255, variable=self.ThresholdingValue , orient="horizontal")
        self.thresholdingScale.grid(row=2, column=0, columnspan=3)
        
        self.thresholdingAcceptButton = tkinter.Button(master=self.thresholding_label, text="Zatwierdź", command=self.applyThresholding)
        self.thresholdingAcceptButton.grid(row=3, column=0)

        self.thresholdingPreviewButton = tkinter.Button(master=self.thresholding_label, text="Podgląd", command=self.previewThresholding)
        self.thresholdingPreviewButton.grid(row=3, column=1)

        self.thresholdingCancelButton = tkinter.Button(master=self.thresholding_label, text="Anuluj", command=self.thresholdingMenu.destroy)
        self.thresholdingCancelButton.grid(row=3, column=2)

        self.thresholding_label.pack()

        del imageRatio
        
    def previewThresholding(self):
        if(self.thresholdingPreview == None): return
        self.previewEditedImage = self.program.thresholding(self.ThresholdingValue.get(), self.checkThresholding.get(), self.program.image.resize((self.sizeThresholding), Image.LANCZOS))
        view = ImageTk.PhotoImage(self.previewEditedImage, Image.LANCZOS)

        self.thresholdingPreview['image'] = view
        self.thresholdingPreview.image = view

        del view
        # self.thresholdingPreview.image = ImageTk.PhotoImage(self.previewEditedImage, Image.LANCZOS)

    def applyThresholding(self):
        if self.program.image == None: return None

        self.thresholdingMenu.destroy()

        self.program.image = self.program.thresholding(self.ThresholdingValue.get(), self.checkThresholding.get(), self.program.image)
        self.oknoUpdate()
        self.updateThumbnail(False, self.autoRefresh.get())

        del self.sizeThresholding, self.checkThresholding, self.ThresholdingValue, self.previewEditedImage
    
    def doPosterization(self):
        def preview():
            posterizationImagePreview = ImageTk.PhotoImage(self.program.posterizationPreview(self.posterizationImage, PosterizationValue.get()))
            posterizationPreview['image'] = posterizationImagePreview
            posterizationPreview.image = posterizationImagePreview
        def accept():
            self.program.applyPosterization(PosterizationValue.get())
            posterizationMenu.destroy()
            del self.posterizationImage

        PosterizationValue = tkinter.IntVar()
        PosterizationValue.set(255)

        # Okienko
        posterizationMenu = tkinter.Toplevel()
        posterizationMenu.title("Posteryzacja")
        posterizationMenu.resizable(False, False)
        posterizationMenu.grab_set()
        posterizationMenu.focus()

        size = (300, 300)
        imageRatio = self.program.image.width/self.program.image.height

        if (1 > imageRatio):
            size = (int(self.program.image.width * (300/self.program.image.height)), 300)
        else:
            size = (300, int(self.program.image.height * (300/self.program.image.width)))

        self.posterizationImage = self.program.image.resize(size, Image.LANCZOS)
        
        posterizationImagePreview = ImageTk.PhotoImage(self.posterizationImage, Image.LANCZOS)
        posterization_label = tkinter.LabelFrame(master=posterizationMenu, padx=10, pady=10, text="Progowanie")
        posterization_label.grid(row=0, column=0)
        posterizationPreview = tkinter.Label(master=posterization_label, image=posterizationImagePreview)
        posterizationPreview.image = posterizationImagePreview
        
        posterizationPreview.grid(row=0, column=0, columnspan=3)
        posterizationScale = tkinter.Scale(master=posterization_label, length=300, from_=2, to=255, variable=PosterizationValue , orient="horizontal")
        posterizationScale.grid(row=2, column=0, columnspan=3)
        posterizationAcceptButton = tkinter.Button(master=posterization_label, text="Zatwierdź", command=accept)
        posterizationAcceptButton.grid(row=3, column=0)

        posterizationPreviewButton = tkinter.Button(master=posterization_label, text="Podgląd", command=preview)
        posterizationPreviewButton.grid(row=3, column=1)

        posterizationCancelButton = tkinter.Button(master=posterization_label, text="Anuluj", command=posterizationMenu.destroy)
        posterizationCancelButton.grid(row=3, column=2)

    def doSmoothing(self):
        if self.program.image is None: return

        def preview():
            if(self.smoothMode==1):
                previewImage = self.program.gaussianBlur(self.smoothBorder.get())
            else:
                previewImage = self.program.blur(self.smoothBorder.get())

            size = (300, 300)
            imageRatio = previewImage.width/previewImage.height

            if (1 > imageRatio):
                size = (int(previewImage.width * (size[1]/previewImage.height)), size[0])
            else:
                size = (size[0], int(previewImage.height * (size[1]/previewImage.width)))
            previewImage = ImageTk.PhotoImage(previewImage.resize(size), Image.LANCZOS)

            smoothingImage['image'] = previewImage
            smoothingImage.image = previewImage

            del previewImage

        def apply():
            if(self.smoothMode==1):
                previewImage = self.program.gaussianBlur(self.smoothBorder.get())
            else:
                previewImage = self.program.blur(self.smoothBorder.get())

            self.program.image = previewImage
            del previewImage
            smoothingMain.destroy()

        smoothMode = tkinter.IntVar()
        smoothingMain = tkinter.Toplevel()
        smoothingMain.title("Operacja wygładzania")
        smoothingMain.geometry("+%d+%d" % (self.okno.winfo_x(), self.okno.winfo_y()))
        smoothingMain.resizable(False, False)
        smoothingMain.grab_set()
        smoothingMain.focus()

        smoothingMenu = tkinter.Label(smoothingMain)
        # Photo ======
        smoothingImagePreview = self.program.image
        # =========
        size = (300, 300)
        imageRatio = smoothingImagePreview.width/smoothingImagePreview.height

        if (1 > imageRatio):
            size = (int(smoothingImagePreview.width * (size[1]/smoothingImagePreview.height)), size[0])
        else:
            size = (size[0], int(smoothingImagePreview.height * (size[1]/smoothingImagePreview.width)))
        # =========
        smoothingImagePreview = ImageTk.PhotoImage(smoothingImagePreview.resize(size), Image.LANCZOS)
        
        smoothingImage = tkinter.Label(smoothingMenu, image=smoothingImagePreview) 
        smoothingImage['image'] = smoothingImagePreview
        smoothingImage.image = smoothingImagePreview
        # ============
        smoothingImage.grid(row=0,columnspan=6)
        smoothingModeText = tkinter.Label(smoothingMenu, text="Typ wygładzania")
        smoothingModeText.grid(row=1, columnspan=6)
        smoothingGaussian = tkinter.Radiobutton(smoothingMenu, text="Gaussian Blur", variable=self.smoothMode, value=0)
        smoothingGaussian.grid(row=2,column=0, columnspan=3)
        smoothingBlur = tkinter.Radiobutton(smoothingMenu, text="Blur", variable=self.smoothMode, value=1)
        smoothingBlur.grid(row=2,column=4, columnspan=3)
        smoothingModeText = tkinter.Label(smoothingMenu, text="Typ granicy (border type)")
        smoothingModeText.grid(row=3, columnspan=6)
        smoothingReflect = tkinter.Radiobutton(smoothingMenu, text="Reflect", variable=self.smoothBorder, value=cv.BORDER_REFLECT)
        smoothingReflect.grid(row=4, column=0, columnspan=2)
        smoothingReplicate = tkinter.Radiobutton(smoothingMenu, text="Replicate", variable=self.smoothBorder, value=cv.BORDER_REPLICATE)
        smoothingReplicate.grid(row=4, column=2, columnspan=2)
        smoothingIsolated = tkinter.Radiobutton(smoothingMenu, text="Isolated", variable=self.smoothBorder, value=cv.BORDER_ISOLATED)
        smoothingIsolated.grid(row=4, column=4, columnspan=2)
        smoothigApply = tkinter.Button(smoothingMenu, text="Zastosuj", command=apply)
        smoothigApply.grid(row=5, column=0, columnspan=2)
        smoothigPreview = tkinter.Button(smoothingMenu, text="Podgląd", command=preview)
        smoothigPreview.grid(row=5, column=2, columnspan=2)
        smoothigCancel = tkinter.Button(smoothingMenu, text="Anuluj", command=smoothingMain.destroy)
        smoothigCancel.grid(row=5, column=4, columnspan=2)
        smoothingMenu.pack()

    def morphologyMainImage(self):
        self.program.image = self.morphologyFunction[self.morphologySize]

    def morphologyApplyImage(self):
        if self.morphologyImagePreview is None: return
        if self.program.image is None: return

        self.morphologyFunction = [self.program.erosion(self.morphologySize, self.program.image, self.morphologyShape.get()),
                            self.program.dilation(self.morphologySize, self.program.image, self.morphologyShape.get()),
                            self.program.opening(self.morphologySize, self.program.image, self.morphologyShape.get()),
                            self.program.closing(self.morphologySize, self.program.image, self.morphologyShape.get())]

        self.program.image = self.morphologyFunction[self.morphologyMode.get()]

        self.morphologyMenu.destroy()

    def morphologyPreviewImage(self):
        if self.morphologyImagePreview is None: return
        if self.program.image is None: return

        imageMorphology = self.program.image

        self.morphologySize = int(self.morphologyFirstSpinboxSize.get())
        
        self.morphologyFunction = [self.program.erosion(self.morphologySize, imageMorphology, self.morphologyShape.get()),
                                    self.program.dilation(self.morphologySize, imageMorphology, self.morphologyShape.get()),
                                    self.program.opening(self.morphologySize, imageMorphology, self.morphologyShape.get()),
                                    self.program.closing(self.morphologySize, imageMorphology, self.morphologyShape.get())]
        
        try:
            imageMorphology = self.morphologyFunction[self.morphologyMode.get()]


            size = (500, 500)
            imageRatio = imageMorphology.width/imageMorphology.height

            if (1 > imageRatio):
                size = (int(imageMorphology.width * (size[1]/imageMorphology.height)), size[0])
            else:
                size = (size[0], int(imageMorphology.height * (size[1]/imageMorphology.width)))
            
            imageDisplay = ImageTk.PhotoImage(imageMorphology.resize((size), Image.LANCZOS))


            self.morphologyImagePreview['image'] = imageDisplay
            self.morphologyImagePreview.image = imageDisplay
            
            del imageDisplay, imageRatio
        except:
            return

    def doArgument(self):
        if self.program.image is None: return

        def checkSecondImage():
            # Włącz możliwość wciśnięcia przycisku po otwarciu drugiego zdjęcia
            if self.program.secondImage is None:
                argumentAdd['state'] = "disabled"
                argumentBlend['state'] = "disabled"
                argumentAND['state'] = "disabled"
                argumentOR['state'] = "disabled"
                argumentNOT['state'] = "disabled"
                argumentXOR['state'] = "disabled"
            else:
                argumentAdd['state'] = "normal"
                argumentBlend['state'] = "normal"
                argumentAND['state'] = "normal"
                argumentOR['state'] = "normal"
                argumentNOT['state'] = "normal"
                argumentXOR['state'] = "normal"

        def Add(refreshHistogram=False):
            first = PillowToCV(self.program.image.convert('RGB'))
            second = PillowToCV(self.program.secondImage.convert('RGB').resize((self.program.image.width, self.program.image.height), Image.LANCZOS))
            self.program.image = CVToPillow(cv.add(first, second))
            argumentMenu.destroy()
            self.updateThumbnail(False, self.autoRefresh.get())
        def Blend(refreshHistogram=False):
            first = PillowToCV(self.program.image.convert('RGB'))
            second = PillowToCV(self.program.secondImage.convert('RGB').resize((self.program.image.width, self.program.image.height), Image.LANCZOS))
            self.program.image = CVToPillow(cv.addWeighted(first, 0.5, second, 0.5, 0.0))
            argumentMenu.destroy()
            self.updateThumbnail(False, self.autoRefresh.get())
        def And(refreshHistogram=False):
            first = PillowToCV(self.program.image.convert('RGB'))
            second = PillowToCV(self.program.secondImage.convert('RGB').resize((self.program.image.width, self.program.image.height), Image.LANCZOS))
            self.program.image = CVToPillow(cv.bitwise_and(first, second))
            argumentMenu.destroy()
            self.updateThumbnail(False, self.autoRefresh.get())
        def Or(refreshHistogram=False):
            first = PillowToCV(self.program.image.convert('RGB'))
            second = PillowToCV(self.program.secondImage.convert('RGB').resize((self.program.image.width, self.program.image.height), Image.LANCZOS))
            self.program.image = CVToPillow(cv.bitwise_or(first, second))
            argumentMenu.destroy()
            self.updateThumbnail(False, self.autoRefresh.get())
        def Not(refreshHistogram=False):
            first = PillowToCV(self.program.image.convert('RGB'))
            second = PillowToCV(self.program.secondImage.convert('RGB').resize((self.program.image.width, self.program.image.height), Image.LANCZOS))
            self.program.image = CVToPillow(cv.bitwise_not(first, second))
            argumentMenu.destroy()
            self.updateThumbnail(False, self.autoRefresh.get())
        def Xor(refreshHistogram=False):
            first = PillowToCV(self.program.image.convert('RGB'))
            second = PillowToCV(self.program.secondImage.convert('RGB').resize((self.program.image.width, self.program.image.height), Image.LANCZOS))
            self.program.image = CVToPillow(cv.bitwise_xor(first, second))
            argumentMenu.destroy()
            self.updateThumbnail(False, self.autoRefresh.get())

        def argumentOpen():
            
            filename = filedialog.askopenfilename(title = "Wybierz obraz", filetypes = (
            ("Obraz JPG","*.jpg;*.jpeg;.jpe;.jif;.jfif;.jfi"),
            ("Obraz PNG","*.png"),
            ("Obraz BMP","*.bmp"),
            ("Wszystkie pliki","*.*"))
            ) #Zapytaj, jakie zdjęcie ma być otwarte
            if filename != "": # Przypadek, gdy użytkownik wybierze zdjęcie
                try:
                    if self.program.secondImage is not None:
                        self.program.secondImage.close()
                    self.program.secondImage = Image.open(filename)
                except:
                    messagebox.showerror("Błąd", "Nie zidentyfikowano zdjęcia!")
                    return self.program.secondImage
            else:
                return

            size = (400, 400)
            imageRatio = self.program.secondImage.width/self.program.secondImage.height

            if (1 > imageRatio):
                size = (int(self.program.secondImage.width * (size[1]/self.program.secondImage.height)), size[0])
            else:
                size = (size[0], int(self.program.secondImage.height * (size[1]/self.program.secondImage.width)))
            
            viewImage = ImageTk.PhotoImage(self.program.secondImage.resize((size), Image.LANCZOS))

            argumentSecond['image'] = viewImage
            argumentSecond.image = viewImage

            checkSecondImage()

        argumentMenu = tkinter.Toplevel()
        argumentMenu.title("Operacje dwuargumentowe")
        argumentMenu.geometry("+%d+%d" % (self.okno.winfo_x(), self.okno.winfo_y()))
        argumentMenu.resizable(False, False)
        argumentMenu.grab_set()
        argumentMenu.focus()

        size = (400, 400)
        imageRatio = self.program.image.width/self.program.image.height

        if (1 > imageRatio):
            size = (int(self.program.image.width * (size[1]/self.program.image.height)), size[0])
        else:
            size = (size[0], int(self.program.image.height * (size[1]/self.program.image.width)))
        
        imageDisplay = ImageTk.PhotoImage(self.program.image.resize((size), Image.LANCZOS))

        #img = ImageTk.PhotoImage(self.program.image.resize())
        argumentFirst = tkinter.Label(argumentMenu, image=imageDisplay)
        argumentFirst.image = imageDisplay
        argumentFirst.grid(row=0,column=0)
        
        secondImage = None
        argumentSecond = tkinter.Label(argumentMenu, image=ImageTk.PhotoImage(Image.new("RGBA", size, (255, 255, 255, 0))))
        argumentSecond.grid(row=0,column=1)
        argumentOperation = tkinter.Label(argumentMenu)
        
        argumentSecondImage = tkinter.Button(argumentMenu, text="Wybierz z dysku zdjęcie", command=argumentOpen)
        argumentSecondImage.grid(row=1, columnspan=2)

        argumentDoText = tkinter.Label(argumentMenu, text="Wykonaj operacje")
        argumentDoText.grid(row=2, columnspan=2)
        argumentAdd = tkinter.Button(argumentMenu, text="Dodawanie", command=Add)
        argumentAdd.grid(row=3, column=0)
        argumentBlend = tkinter.Button(argumentMenu, text="Mieszanie", command=Blend)
        argumentBlend.grid(row=3, column=1)
        argumentAND = tkinter.Button(argumentMenu, text="And", command=And)
        argumentAND.grid(row=4, column=0)
        argumentOR = tkinter.Button(argumentMenu, text="Or", command=Or)
        argumentOR.grid(row=4, column=1)
        argumentNOT = tkinter.Button(argumentMenu, text="Not", command=Not)
        argumentNOT.grid(row=5, column=0)
        argumentXOR = tkinter.Button(argumentMenu, text="Xor", command=Xor)
        argumentXOR.grid(row=5, column=1)

        argumentOperation.grid(row=1,column=0, columnspan=2)
        checkSecondImage()

    def doMorphology(self):
        if self.program.image is None: return

        def morphologySecondFilterSwitch():
            if(self.SecondFilterCheck.get()):
                stateVal = "normal"
            else:
                stateVal = "disabled"

            morphologySecondErosionOption['state'] = stateVal
            morphologySecondDilatationOption['state'] = stateVal
            morphologySecondOpeningOption['state'] =  stateVal
            morphologySecondClosingOption['state'] = stateVal
            morphologySecondMaskText['state'] = stateVal
            self.morphologySecondSpinboxSize['state'] = stateVal

            del stateVal

        self.morphologyOriginalImage = self.program.image
        self.morphologyElementStructure = tkinter.IntVar()

        self.morphologyMenu = tkinter.Toplevel()
        self.morphologyMenu.title("Morfologia")
        self.morphologyMenu.geometry("+%d+%d" % (self.okno.winfo_x(), self.okno.winfo_y()))
        self.morphologyMenu.resizable(False, False)
        self.morphologyMenu.grab_set()
        self.morphologyMenu.focus()

        morphologyMainLabel = tkinter.Label(self.morphologyMenu, width=250, height=250)

        # Row 0 ; Column 0
        morphologyPreview = tkinter.LabelFrame(morphologyMainLabel, width=300, height=200, text="Podgląd")
        self.morphologyImagePreview = tkinter.Label(morphologyPreview, image=ImageTk.PhotoImage(Image.new('RGBA',(500, 500), (255, 255, 255, 0))))
        self.morphologyImagePreview.pack()
        morphologyPreview.grid(row=0, column=0, sticky="NS")

        # Row 0 ; Column 1
        morphologyMode = tkinter.LabelFrame(morphologyMainLabel, text="Wybór morfologii")
        #  Radiobutton - opcje

        # Pierwsza opcja
        morphologyFirstFilterFrame = tkinter.LabelFrame(morphologyMode)
        morphologyFirstErosionOption = tkinter.Radiobutton(morphologyFirstFilterFrame, text="Erozja", variable=self.morphologyMode, value=0)
        morphologyFirstErosionOption.grid(row=0, columnspan=2, sticky = 'W')
        morphologyFirstDilatationOption = tkinter.Radiobutton(morphologyFirstFilterFrame, text="Dylatacja", variable=self.morphologyMode, value=1)
        morphologyFirstDilatationOption.grid(row=1, columnspan=2, sticky = 'W')
        morphologyFirstOpeningOption = tkinter.Radiobutton(morphologyFirstFilterFrame, text="Otwarcie", variable=self.morphologyMode, value=2)
        morphologyFirstOpeningOption.grid(row=2, columnspan=2, sticky = 'W')
        morphologyFirstClosingOption = tkinter.Radiobutton(morphologyFirstFilterFrame, text="Zamknięcie", variable=self.morphologyMode, value=3)
        morphologyFirstClosingOption.grid(row=3, columnspan=2, sticky = 'W')
        morphologyFirstMaskText = tkinter.Label(morphologyFirstFilterFrame, text="Rozmiar maski ")
        morphologyFirstMaskText.grid(row=4, column=0, sticky='W')
        self.morphologyFirstSpinboxSize = tkinter.Spinbox(morphologyFirstFilterFrame, from_=3, to=25, increment=2, width=4) #
        self.morphologyFirstSpinboxSize.grid(row=4, column=1)
        morphologyFirstFilterFrame.grid(row=1, sticky="EW")
        
        morphologyShapeLabel = tkinter.LabelFrame(morphologyMode, text="Element strukturalny")
        morphologyShapeRect = tkinter.Radiobutton(morphologyShapeLabel, text="Kwadrat", variable=self.morphologyShape, value=0) #
        morphologyShapeRect.grid(row=0, sticky="W")
        morphologyShapeRhombus = tkinter.Radiobutton(morphologyShapeLabel, text="Romb", variable=self.morphologyShape, value=1) #
        morphologyShapeRhombus.grid(row=1, sticky="W")
        morphologyShapeLabel.grid(row=2, sticky="EW")

        morphologyMode.grid(row=0, column=1, sticky="NS")

        # Row 1 ; Column 1

        morphologyUnderMode = tkinter.Label(morphologyMainLabel)
        morphologyPreviewButton = tkinter.Button(morphologyUnderMode, text="Podgląd", command=self.morphologyPreviewImage)
        morphologyPreviewButton.grid(row=0, column=0)
        
        morphologyAcceptButton = tkinter.Button(morphologyUnderMode, text="Wykonaj", comman=self.morphologyApplyImage)
        morphologyAcceptButton.grid(row=0, column=1)

        morphologyUnderMode.grid(row=1, columnspan=2)
        
        morphologyMainLabel.pack()

        self.morphologyPreviewImage()

    def doWatersheding(self):
        if self.program.image is None: return

        self.watershedFullScreen = None
        self.watershedStep = None

        stepTitles = ("Zamiana obraz na odcień szaro odcieniowy",
                      "Wykonywanie progowania binarnego",
                      "Wykonanie operacji otwierania do usunięcia małych obiektów",
                      "Wyznacz obszar tła, używając dylacji",
                      "Transformacja odległościowa (wizualizacja)",
                      "Stwórz skupisko obiektów",
                      "Działanie odejmowania, określające nakrycie się obu obiektów",
                      "Etykietowanie obiektów",
                      "Dodanie etykiety dla ułatwienia działania algorytmu wodociągu",
                      "Algorytm wodociągu - efekt końcowy")

        self.watershedingOriginalImage = self.program.image

        def startWatersheding():
            image, count = self.program.watershed()
            self.watershedFullScreen = image.copy()
            image = CVToPillow(image)
            image.thumbnail((1000, 1000))
            imageView = ImageTk.PhotoImage(image)

            self.watershedingTextCount['text'] = "Liczba obiektów: " + str(count)
            self.watershedingTextCount.text = "Liczba obiektów: " + str(count)
            self.watershedingImagePreview['image'] = imageView
            self.watershedingImagePreview.image = imageView

        def stepByStep():
            def previousSteps():
                if(self.watershedPage > 0): 
                    self.watershedPage = self.watershedPage - 1
                    loadSteps()

            def nextSteps():
                if(self.watershedPage < self.watershedAllPages): 
                    self.watershedPage = self.watershedPage + 1
                    loadSteps()
            
            def loadSteps():
                if self.watershedStep is not None:
                    self.watershedStep.destroy()
                self.watershedStep = tkinter.Label(watershedStepsMain, text="")
                self.watershedStep.grid(row=0, column=0, columnspan=2, sticky="W")
                self.waterizetextInfo.set("Krok " + str(self.watershedPage+1) + " - " + stepTitles[self.watershedPage])
                self.watershedStep['text'] = self.waterizetextInfo.get()
                self.watershedStep.text = self.waterizetextInfo.get()
                showImage = Image.fromarray(stepsList[self.watershedPage])
                watershedImage = ImageTk.PhotoImage(Image.fromarray(stepsList[self.watershedPage]), Image.LANCZOS)
                watershedPreview = tkinter.Label(watershedStepsMain, image=watershedImage)
                watershedPreview.grid(row=1, column=0, columnspan=2)
                watershedPreview['image'] = watershedImage
                watershedPreview.image = watershedImage
                watershedPrevious  = tkinter.Button(watershedStepsMain, text="Wróć", command=previousSteps) 
                watershedPrevious.grid(row=2, column=0)
                watershedNext = tkinter.Button(watershedStepsMain, text="Dalej", command=nextSteps) 
                watershedNext.grid(row=2, column=1)

            watershedSteps = tkinter.Toplevel()
            watershedSteps.title("Wododział")
            watershedSteps.geometry("+%d+%d" % (self.okno.winfo_x(), self.okno.winfo_y()))
            watershedSteps.resizable(False, False)
            watershedSteps.grab_set()
            watershedSteps.focus()

            stepsList = self.program.watershedSteps()
            count = stepsList[:-1]
            del stepsList[-1]
            for i in stepsList:
                i = CVToPillow(i)
            tuple(stepsList)
            self.watershedPage = 0
            self.watershedAllPages = len(stepsList)-1

            watershedStepsMain = tkinter.Label(watershedSteps, width=300, height=400)
            watershedStepsMain.pack()

            
            loadSteps()

        def fullImage():
            if self.watershedFullScreen is None: 
                messagebox.showinfo("Brak zdjęcia", "Zdjęcie nie zostało jeszcze wygenereowane.")
                return None
            cv.imshow("Full preview", self.watershedFullScreen)

        watershedingMenu = tkinter.Toplevel()
        watershedingMenu.title("Metoda wododziałowa")
        watershedingMenu.geometry("+%d+%d" % (self.okno.winfo_x(), self.okno.winfo_y()))
        watershedingMenu.resizable(False, False)
        watershedingMenu.grab_set()
        watershedingMenu.focus()

        watershedingMainLabel = tkinter.Label(watershedingMenu, width=250, height=250)

        # Row 0 ; Column 0
        watershedingPreview = tkinter.LabelFrame(watershedingMainLabel, width=300, height=200, text="Podgląd")
        self.watershedingImagePreview = tkinter.Label(watershedingPreview, image=ImageTk.PhotoImage(Image.new('RGBA',(500, 500), (255, 255, 255, 0))))
        self.watershedingImagePreview.pack()
        watershedingPreview.grid(row=0, column=0, sticky="NS")

        # Row 0 ; Column 1
        watershedingMode = tkinter.LabelFrame(watershedingMainLabel, text="Informacje")
        
        # Liczba obiektów
        self.watershedingTextCount = tkinter.Label(watershedingMode, text="Liczba obiektów:")
        self.watershedingTextCount.grid(row=0)

        watershedingGo = tkinter.Button(watershedingMode, text="Wykonaj", command=startWatersheding)
        watershedingGo.grid(row=1)

        watershedingStep = tkinter.Button(watershedingMode, text="Krok po kroku", command=stepByStep)
        watershedingStep.grid(row=2)

        watershedingFull = tkinter.Button(watershedingMode, text="Pełny obraz", command=fullImage)
        watershedingFull.grid(row=3)

        watershedingMode.grid(row=0, column=1, sticky="NS")
        
        watershedingMainLabel.pack()

# ===================================
#  Program do egzaminu
# ===================================

    def doBackProjection(self):
        if self.program.image is None: return

        def histBack():

            def scaleData(value):
                left, right = self.program.Hist_and_Backproj(leftImage, int(value))
                

                Ltmp = CVToPillow(left)
                Ltmp.thumbnail((300, 300))
                Rtmp = CVToPillow(right)
                Rtmp.thumbnail((300, 300))

                leftView = ImageTk.PhotoImage(Ltmp)
                rightView = ImageTk.PhotoImage(Rtmp)

                del Ltmp, Rtmp

                histBackLeftImage['image'] = leftView
                histBackLeftImage.image = leftView
                histBackRightImage['image'] = rightView
                histBackRightImage.image = rightView


            histBackMain = tkinter.Toplevel()
            histBackMain.resizable(False, False)
            histBackMain.title("Model histogramu")
            histBackMain.grab_set()
            histBackMain.focus()

            histBackMainLabel = tkinter.Label(histBackMain)

            # Inicjalizacja
            leftImage = PillowToCV(self.program.image)
            
            left, right = self.program.Hist_and_Backproj(leftImage, 0)
            left = CVToPillow(left)
            right = CVToPillow(right)
            left.thumbnail((300, 300))
            right.thumbnail((300, 300))

            # Lewa strona
            histBackLeft = tkinter.LabelFrame(histBackMainLabel, text="Efekt metody Back Projection")
            histBackLeftImage = tkinter.Label(histBackLeft)
            histBackLeftImage.pack()
            histBackLeft.grid(row=0, column=0, sticky="NSEW")

            # Prawa strona
            histBackRight = tkinter.LabelFrame(histBackMainLabel, text="Histogram")
            histBackRightImage = tkinter.Label(histBackRight)
            histBackRightImage.pack()
            histBackRight.grid(row=0, column=1, sticky="EW")

            # Wstawianie okienka po inicjalizacji okna obrazkowego
            leftView = ImageTk.PhotoImage(left)
            rightView = ImageTk.PhotoImage(right)

            histBackLeftImage['image'] = leftView
            histBackLeftImage.image = leftView
            histBackRightImage['image'] = rightView
            histBackRightImage.image = rightView

            # Dolna zakładka
            histBackBottom = tkinter.LabelFrame(histBackMainLabel)
            
            # Suwak
            histBackScale = tkinter.Scale(histBackBottom, length=600, from_=0, to=180, orient="horizontal", command=scaleData)
            histBackScale.grid(row=0, column=0, sticky="NSEW")
            
            histBackBottom.grid(row=1, column=0, columnspan=2)

            histBackMainLabel.grid(sticky="NSEW")

        def maskProjection():
            '''Maskowanie obiektu'''
            def pickPoint(event, x, y, flags, param):
                '''Funkcja, która jest wywołana w momencie kliknięcia na obrazek'''
                # Przejdź dalej, jeżeli użytkowni wcisnął lewy przycisk myszy
                if event != cv.EVENT_LBUTTONDOWN:
                    return
                (a, b) = mask.pickPoint(event, x, y, flags, param)
                
                # Generuj na postać możliwy do wyświetlania obrazu
                a = CVToPillow(a)
                b = CVToPillow(b)
                a.thumbnail((500, 500))
                b.thumbnail((500, 500))
                a = ImageTk.PhotoImage(a)
                b = ImageTk.PhotoImage(b)

                maskLeftImage['image'] = a
                maskLeftImage.image = a
                maskRightImage['image'] = b
                maskRightImage.image = b

                cv.destroyWindow('Wybierz punkt')

            def selector():
                '''Wywoływana funkcja po wciśnięciu Pobierz kolor'''
                #mask.selectImagePosition()
                
                # Stwórz nowe okienko, w którym możemy pobrać wartości punktów na podstawie
                # kliknięcia w dowolnym miejscu obrazka
                window_image = 'Wybierz punkt'
                cv.namedWindow(window_image)
                cv.imshow(window_image, mask.pillowImage)

                cv.setMouseCallback(window_image, pickPoint)

            
            maskMain = tkinter.Toplevel()
            maskMain.resizable(False, False)
            maskMain.title("Maskowanie obiektu")
            #maskMain.grab_set()
            maskMain.focus()

            maskMainLabel = tkinter.Label(maskMain)

            # Inicjalizacja klasy maskowania obiektu
            mask = BackProjection_Mask(self.program.image)

            (a, b) = mask.pickPoint(1, 1, 1, 1, None)

            a = CVToPillow(a)
            b = CVToPillow(b)
            a.thumbnail((500, 500))
            b.thumbnail((500, 500))
            a = ImageTk.PhotoImage(a)
            b = ImageTk.PhotoImage(b)

            # Okienko z oryginalnym zdjęciem
            maskOriginal = tkinter.LabelFrame(maskMainLabel, text="Oryginalne zdjęcie")
            maskOriginalImage = tkinter.Label(maskOriginal)
            maskOriginalImage.pack()
            maskOriginal.grid(row=0, column=0, sticky="EW")
            
            # Maskowanie
            maskLeft = tkinter.LabelFrame(maskMainLabel, text="Maskowanie")
            maskLeftImage = tkinter.Label(maskLeft)
            maskLeftImage.pack()
            maskLeft.grid(row=0, column=1, sticky="EW")

            # Prawa strona
            maskRight = tkinter.LabelFrame(maskMainLabel, text="Projekcja wsteczna")
            maskRightImage = tkinter.Label(maskRight)
            maskRightImage.pack()
            maskRight.grid(row=0, column=2, sticky="EW")

            maskBottom = tkinter.LabelFrame(maskMainLabel)
            maskBottom.grid(row=1, column=0, columnspan=3, sticky="NSEW")

            originalImage = self.program.image
            originalImage.thumbnail((500, 500))
            originalView = ImageTk.PhotoImage(originalImage)

            # Wstawianie zdjęć do okienek po inicjalizacji okna obrazkowego
            maskOriginalImage['image'] = originalView
            maskOriginalImage.image = originalView
            maskLeftImage['image'] = a
            maskLeftImage.image = a
            maskRightImage['image'] = b
            maskRightImage.image = b

            # Dolna zakładka
            maskBottom = tkinter.LabelFrame(maskMainLabel)
            maskBottomButton = tkinter.Button(maskBottom, text="Pobierz kolor", width=20, height=2, command=selector)
            maskBottomButton.grid()
            maskBottom.grid(row=1, column=0, columnspan=3)

            maskMainLabel.grid(sticky="NSEW")

            maskMain.grid_columnconfigure(0,weight=1)
            maskMain.grid_columnconfigure(1,weight=1)
            maskMain.grid_rowconfigure(0,weight=1)
            maskMain.grid_rowconfigure(1,weight=1)

        def model():
            backProjectionMenu.destroy()
            histBack()

        def mask():
            backProjectionMenu.destroy()
            maskProjection()

        backProjectionMenu = tkinter.Toplevel()
        backProjectionMenu.title("Projekcja wsteczna")
        backProjectionMenu.resizable(False, False)
        backProjectionMenu.grab_set()
        backProjectionMenu.focus()


        backProjectionMainLabel = tkinter.Label(backProjectionMenu)

        backProjectionInfo = tkinter.Label(backProjectionMainLabel, text="Proszę wybrać tryb projekcji wstecznej")
        backProjectionInfo.grid(row=0, column=0, columnspan=2, pady=20)

        backProjectionHistogram1D = tkinter.Button(backProjectionMainLabel, text="Model histogramu", command=model)
        backProjectionHistogram1D.grid(row=1, column=0, padx=5, pady=5)

        backProjectionMask = tkinter.Button(backProjectionMainLabel, text="Maskowanie obiektu", command=mask)
        backProjectionMask.grid(row=1, column=1, padx=5, pady=5)
        
        backProjectionMainLabel.pack()


        backProjectionMenu.geometry("+%d+%d" % (((backProjectionMenu.winfo_screenwidth()/2) - 126), 
                                                ((backProjectionMenu.winfo_screenheight()/2) - 50)))

# ===================================
#  Koniec treści do egzaminu 
# ===================================

    def GraphSetValue(self):
        self.widthGraphValue = self.graphWidth.get()
        self.heightGraphValue = self.graphHeight.get()

    def bgColorGraphSelect(self):
        color = colorchooser.askcolor(self.bgColorGraphValue)
        self.bgColorGraphValue = color[1]
        self.graphBgColor['bg'] = self.bgColorGraphValue
        try:
            self.optionsMenu.focus_set()
        except:
            pass
        del color

    def clrLineGraphSelect(self):
        color = colorchooser.askcolor(self.clrLineGraphValue)
        self.clrLineGraphValue = color[1]
        self.graphLineColor['bg'] = self.clrLineGraphValue
        try:
            self.optionsMenu.focus_set()
        except:
            pass
        del color
 
    def clrHorizontalGraphSelect(self):
        color = colorchooser.askcolor(self.clrHorizontalGraphValue)
        self.clrHorizontalGraphValue = color[1]
        self.graphHorizontalColor['bg'] = self.clrHorizontalGraphValue
        try:
            self.optionsMenu.focus_set()
        except:
            pass
        del color
            

# ============================================

# ============================================
# Okno zapytaniowe odnośnie histogramu rozciągającego
    def options(self):
        self.optionsMenu = tkinter.Toplevel()
        self.optionsMenu.geometry("+%d+%d" % (self.okno.winfo_x()+320, self.okno.winfo_y()+200))
        self.optionsMenu.title("Opcje")
        self.optionsMenu.resizable(False, False)

        self.optionsMenu.focus_set()
        # =======================
        self.graphOptionMain = tkinter.Label(self.optionsMenu)

        self.graphOption_label = tkinter.LabelFrame(master=self.graphOptionMain, width=400, height=100, padx=10, pady=10, text="Wykres")

        self.graphWidthText = tkinter.Label(self.graphOption_label, text="Długość")
        self.graphWidthText.grid(row=0, column=0, sticky="W")
        
        self.graphWidth = tkinter.Spinbox(self.graphOption_label, from_=328, to=1976, command=self.GraphSetValue)
        self.graphWidth.delete(0, 4)
        self.graphWidth.insert(0, str(self.widthGraphValue))
        self.graphWidth.grid(row=1, column=0, sticky="W")
        
        self.graphWidthInfo = tkinter.Label(self.graphOption_label, text="Min: 328 ; Max: 1976")
        self.graphWidthInfo.grid(row=2, column=0, sticky="W")

        self.graphHeightText = tkinter.Label(self.graphOption_label, text="Szerokość")
        self.graphHeightText.grid(row=0, column=1, sticky="W")
        
        self.graphHeight = tkinter.Spinbox(self.graphOption_label, from_=256, to=1536, command=self.GraphSetValue)
        self.graphHeight.delete(0, 4)
        self.graphHeight.insert(0, str(self.heightGraphValue))
        self.graphHeight.grid(row=1, column=1, sticky="W")

        self.graphHeightInfo = tkinter.Label(self.graphOption_label, text="Min: 256 ; Max: 1526")
        self.graphHeightInfo.grid(row=2, column=1, sticky="W")
        
        # Obraz kolorowy
        self.graphColorOption_label = tkinter.LabelFrame(master=self.graphOption_label, text="Obraz kolorowy")

        self.graphRedHistogramTitleText = tkinter.Label(self.graphColorOption_label, text="Tytuł histogramu czerwonego")
        self.graphRedHistogramTitleText.grid(row=4, column=0, sticky="W")

        self.graphGreenHistogramTitleText = tkinter.Label(self.graphColorOption_label, text="Tytuł histogramu zielonego")
        self.graphGreenHistogramTitleText.grid(row=4, column=1, sticky="W")

        self.graphBlueHistogramTitleText = tkinter.Label(self.graphColorOption_label, text="Tytuł histogramu niebieskiego")
        self.graphBlueHistogramTitleText.grid(row=4, column=2, sticky="W")
        
        self.graphRedHistogramTitle = tkinter.Entry(self.graphColorOption_label)
        self.graphRedHistogramTitle.insert(0, self.RedHistogramTitle)
        self.graphRedHistogramTitle.grid(row=5, column=0, sticky="W")

        self.graphGreenHistogramTitle = tkinter.Entry(self.graphColorOption_label)
        self.graphGreenHistogramTitle.insert(0, self.GreenHistogramTitle)
        self.graphGreenHistogramTitle.grid(row=5, column=1, sticky="W")

        self.graphBlueHistogramTitle = tkinter.Entry(self.graphColorOption_label)
        self.graphBlueHistogramTitle.insert(0, self.BlueHistogramTitle)
        self.graphBlueHistogramTitle.grid(row=5, column=2, sticky="W")

        self.graphColorOption_label.grid(row=4, column=0, columnspan=3, sticky="NSEW")

        # Odcień szarości
        
        self.graphGrayOption_label = tkinter.LabelFrame(master=self.graphOption_label, text="Obraz szaro odcieniowy")

        self.graphGrayHistogramTitleText = tkinter.Label(self.graphGrayOption_label, text="Tytuł histogramu czarno-białego")
        self.graphGrayHistogramTitleText.grid(row=0, column=0, columnspan=3, sticky="W")
        
        self.graphGrayHistogramTitle = tkinter.Entry(self.graphGrayOption_label)
        self.graphGrayHistogramTitle.insert(0, self.GrayHistogramTitle)
        self.graphGrayHistogramTitle.grid(row=1, column=0, columnspan=3, sticky="W")

        self.graphGrayOption_label.grid(row=5, column=0, columnspan=3, sticky="NSEW")

        self.graphOption_label.grid(row=0, column=0, columnspan=2)

        self.optionSave = tkinter.Button(self.graphOptionMain, text="Zapisz", command=self.saveOption)
        self.optionSave.grid(row=1, column=0, sticky="E")
        self.optionCancel = tkinter.Button(self.graphOptionMain, text="Anuluj", command=self.optionsMenu.destroy)
        self.optionCancel.grid(row=1, column=1, sticky="W")
        self.graphOptionMain.pack()

# ============================================

    def saveOption(self):
        self.widthGraphValue = int(self.graphWidth.get())
        self.heightGraphValue = int(self.graphHeight.get())
        self.RedHistogramTitle = self.graphRedHistogramTitle.get()
        self.GreenHistogramTitle = self.graphGreenHistogramTitle.get()
        self.BlueHistogramTitle = self.graphBlueHistogramTitle.get()
        self.GrayHistogramTitle = self.graphGrayHistogramTitle.get()
        self.optionsMenu.destroy()

# back = Program()
# back.backProjection()
action = Window()
