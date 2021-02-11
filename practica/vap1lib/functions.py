import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def adjustIntensity(image, inRange=[], outRange=[0.0, 1.0]):
  """
  :param image: input image format NxM
  :param inRange: in range to adjust 1x2
  :paran outRange: out range to adjust 1x2
  :return: image adjusted
  """
  if not inRange :
    inRange = [np.min(image), np.max(image)]

  imin, imax = (inRange[0], inRange[1])
  omin, omax = (outRange[0], outRange[1])

  image = np.clip(image, imin, imax)

  if imin != imax:
    image = (image - imin) / (imax - imin)
    image = image * (omax - omin) + omin
    return np.asarray(image)
  
  return np.clip(image, omin, omax)

def equalizeIntensity(image, nBins=256):
  hist, bin_edges = np.histogram(image, bins=nBins, range=None)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2. 
  img_cdf = hist.cumsum()
  img_cdf = img_cdf / float(img_cdf[-1])
  out = np.interp(image.flat, bin_centers, img_cdf)
  return out.reshape(image.shape)

def filterImage(inImage, kernel):
  (imgX, imgY) = inImage.shape
  copiaI = np.zeros((imgX, imgY))

  try: 
    (kerX, kerY) = kernel.shape
  except:
    try:
      kernel = np.array([kernel])
      (kerX, kerY) = kernel.shape
    except:
      raise Exception("Kernel shape error ",kernel.shape)

  kCenter = (int(np.floor(kerX/2)), int(np.floor(kerY/2)))

  copiaB = np.zeros((kerX+imgX-1, kerY+imgY-1))
  copiaB[kCenter[0]:kCenter[0]+imgX, kCenter[1]:kCenter[1]+imgY] = inImage

  for x in range(0, imgX):
    for y in range(0, imgY):
      copiaI[x,y] = (kernel*copiaB[x:x+kerX, y:y+kerY]).sum()

  return  copiaI

def gaussKernel1D(sigma):
  N = np.ceil(3*sigma)
  x = np.arange(-int(N), int(N+1), 1) 
  p2 = np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))
  p1 = 1/(np.sqrt(2 * np.pi)*sigma)

  plt.plot(x,p1*p2)
  plt.show()

  return p1*p2

def gaussianFilter(inImage, sigma):
  kt = np.array([gaussKernel1D(sigma)])
  k = kt.T
  return filterImage(filterImage(inImage, k), kt)

def medianFilter(inImage, filterSize):

  if filterSize % 2 == 0: 
    raise Exception("filterSize shape error ",filterSize)

  mKernel = np.ones((filterSize, filterSize)) / (filterSize * filterSize)
  
  return filterImage(inImage, mKernel)

def highBoost(inImage, A, method, param):
  copiaI = np.copy(inImage)

  hb_method_types = ['gaussian', 'median']
  hb_mehtod = {
      hb_method_types[0] : gaussianFilter,
      hb_method_types[1] : medianFilter
  }

  if method not in hb_method_types:
    raise Exception("not valid Method")
  
  f_smooth = hb_mehtod.get(method)(copiaI, param)
  sustracted = (A*copiaI - f_smooth)
  
  return sustracted

def gray_scale_erode(inImage, SE, center=[]):
  (ix, iy) = inImage.shape
  
  try:
    (p,q) = SE.shape
  except:
    try:
      (p,) = SE.shape
      q = 1
    except:
      raise Exception("Error wrong SE shape ",SE.shape)

  if not center:
    (cx, cy) = (int(np.floor(p/2)), int(np.floor(q/2)))
  else:
    (cx, cy) = (center[0], center[1])

  copiaB = np.zeros((p+ix-1, q+iy-1))
  copiaB[cx:cx+ix, cy:cy+iy] = inImage

  for x in range(0, ix):
    for y in range(0, iy):
      copiaB[x,y] = np.min(copiaB[x:x+p, y:y+q]-SE)
        
  return copiaB[cx:cx+ix, cy:cy+iy]

def gray_scale_dilate(inImage, SE, center=[]):
  (ix, iy) = inImage.shape
  
  try:
    (p,q) = SE.shape
  except:
    try:
      (p,) = SE.shape
      q = 1
    except:
      raise Exception("Error wrong SE shape ",SE.shape)

  if not center:
    (cx, cy) = (int(np.floor(p/2)), int(np.floor(q/2)))
  else:
    (cx, cy) = (center[0], center[1])

  copiaB = np.zeros((p+ix-1, q+iy-1))
  copiaC = np.zeros((p+ix-1, q+iy-1))
  copiaB[cx:cx+ix, cy:cy+iy] = inImage

  for x in range(0, ix):
    for y in range(0, iy):
      copiaC[x,y] = np.max(copiaB[x:x+p, y:y+q]+SE)
        
  return copiaC[cx:cx+ix, cy:cy+iy]


def erode(inImage, SE, center=[]):
  (ix, iy) = inImage.shape
  
  try:
    (p,q) = SE.shape
  except:
    try:
      (p,) = SE.shape
      q = 1
    except:
      raise Exception("Error wrong SE shape ",SE.shape)

  if not center:
    (cx, cy) = (int(np.floor(p/2)), int(np.floor(q/2)))
  else:
    (cx, cy) = (center[0], center[1])

  copiaB = np.zeros((p+ix-1, q+iy-1))
  copiaC = np.zeros((p+ix-1, q+iy-1))
  copiaB[cx:cx+ix, cy:cy+iy] = inImage

  for x in range(0, ix):
    for y in range(0, iy):
      if inImage[x,y]==0.0 :
        copiaC[x:x+p, y:y+q] = np.minimum((1.0-SE), copiaB[x:x+p, y:y+q])
        
  return copiaC[cx:cx+ix, cy:cy+iy]

def dilate(inImage, SE, center=[]):
  (ix, iy) = inImage.shape
  
  try:
    (p,q) = SE.shape
  except:
    try:
      (p,) = SE.shape
      q = 1
    except:
      raise Exception("Error wrong SE shape ",SE.shape)

  if not center:
    (cx, cy) = (int(np.floor(p/2)), int(np.floor(q/2)))
  else:
    (cx, cy) = (center[0], center[1])

  copiaB = np.zeros((p+ix-1, q+iy-1))
  copiaB[cx:cx+ix, cy:cy+iy] = inImage

  for x in range(0, ix):
    for y in range(0, iy):
      if inImage[x,y]==1.0 :
        copiaB[x:x+p, y:y+q] = np.maximum(SE,copiaB[x:x+p, y:y+q])
        
  return copiaB[cx:cx+ix, cy:cy+iy]

def opening (inImage, SE, center=[]):
  return dilate(erode(inImage, SE, center),SE, center)

def closing (inImage, SE, center=[]):
  return erode(dilate(inImage, SE, center),SE, center)

def fill (inImage, seeds, SE=[], center=[]):
  A = inImage
  end = False

  if not SE: SE = np.array([[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]])
  else: SE = np.array(SE)
  if not center:center = [1,1]
  
  for p in seeds :
    A_c = 1.0 - A
    X = np.zeros(inImage.shape)
    X[p] = 1.0
    while not end:
      X_p = np.minimum(dilate(X,SE,center), A_c)
      if (X_p == X).all(): end = True
      X = X_p
    A = np.maximum(X_p, A)
    end = False
  
  return A

def conV1(inImage, kernel, centers=()):
  (imgX, imgY) = inImage.shape
  copiaI = np.zeros((imgX, imgY))
  
  try: 
    (kerX, kerY) = kernel.shape
  except:
    try:
      kernel = np.array([kernel])
      (kerX, kerY) = kernel.shape
    except:
      raise Exception("Kernel shape error ",kernel.shape)

  if not centers:
    kCenter = (int(np.floor(kerX/2)), int(np.floor(kerY/2)))
  else:
    kCenter = centers

  copiaB = np.zeros((kerX+imgX-1, kerY+imgY-1))
  copiaB[kCenter[0]:kCenter[0]+imgX, kCenter[1]:kCenter[1]+imgY] = inImage

  for x in range(0, imgX):
    for y in range(0, imgY):
      copiaI[x,y] = (kernel*copiaB[x:x+kerX, y:y+kerY]).sum()

  return copiaI  

def f_sobel(inImage, f_conv):
  conv_1_x = f_conv(inImage, np.array([1,2,1]).reshape(3,1) )
  x_gradi  = f_conv(conv_1_x, np.array([-1,0,1]).reshape(1,3) )

  conv_1_y = f_conv(inImage, np.array([1,0,-1]).reshape(3,1) )
  y_gradi  = f_conv(conv_1_y, np.array([1,2,1]).reshape(1,3) )

  return [x_gradi, y_gradi] 

def f_prewitt(inImage, f_conv):
  conv_1_x = f_conv(inImage, np.array([1,1,1]).reshape(3,1) )
  x_gradi  = f_conv(conv_1_x, np.array([-1,0,1]).reshape(1,3) )

  conv_1_y = f_conv(inImage, np.array([1,0,-1]).reshape(3,1) )
  y_gradi  = f_conv(conv_1_y, np.array([1,1,1]).reshape(1,3) )

  return [x_gradi, y_gradi] 

def f_centralDiff(inImage, f_conv):
  x_gradi = f_conv(inImage, np.array([[-1,0,1]]))
  y_gradi = f_conv(inImage, np.array([[-1,0,1]]).T)

  return [x_gradi, y_gradi] 

def f_robert(inImage, f_conv):
  x_gradi = f_conv(inImage, np.array([[-1,0],[0,1]]))
  y_gradi = f_conv(inImage, np.array([[0,-1],[1,0]]))

  return [x_gradi, y_gradi] 

def gradientImage(inImage, operador):
  op_types = ['Roberts', 'CentralDiff','Prewitt','Sobel']
  op_functions = {
      op_types[0] : f_robert,
      op_types[1] : f_centralDiff,
      op_types[2] : f_prewitt,
      op_types[3] : f_sobel
  }

  if operador not in op_types:
    raise Exception("Wrong op, ",operador, " not in ", op_types)
  
  op_function= op_functions.get(operador)

  return op_function(inImage, conV1)

def asignNewValue(bins, value, x, y, Em):
  (xMax, yMax) = Em.shape
  (xMax, yMax) = (xMax-1, yMax-1)
  Im = Em
  (itX, itY) = (x,y)

  maximo = Em[x,y]
  (mX,mY) = (itX,itY)

  if value >= bins[0] and value < bins[1]:
    itY =  itY + 1
    while Em[itX, itY] != 0 and itY < yMax:
      if maximo > Em[itX, itY]:
        Im[itX, itY] = 0
      else:
        maximo = Em[itX, itY]
        Im[mX, mY] = 0
        (mX,mY) = (itX,itY)
      itY =  itY + 1
    
  elif value >= bins[1] and value < bins[2]:
    (itX, itY) = (itX + 1, itY + 1) 
    while Em[itX, itY] != 0 and itX < xMax and itY < yMax:
      if maximo > Em[itX, itY]:
        Im[itX, itY] = 0
      else:
        maximo = Em[itX, itY]
        Im[mX, mY] = 0
        (mX,mY) = (itX,itY)
      (itX, itY) = (itX + 1, itY + 1) 

  elif value >= bins[2] and value < bins[3]:
    (itX, itY) = (itX + 1, itY) 
    while Em[itX, itY] != 0 and itX < xMax:
      if maximo > Em[itX, itY]:
        Im[itX, itY] = 0
      else:
        maximo = Em[itX,itY]
        Im[mX, mY] = 0
        (mX,mY) = (itX,itY)
      (itX, itY) = (itX + 1, itY) 
    
  elif value >= bins[3] and value < bins[4]:
    (itX, itY) = (itX + 1, itY - 1)
    while Em[itX, itY] != 0 and itX < xMax and itY>=0:
      if maximo > Em[itX, itY]:
        Im[itX, itY] = 0
      else:
        maximo = Em[itX,itY]
        Im[mX, mY] = 0
        (mX,mY) = (itX,itY)
      (itX, itY) = (itX + 1, itY - 1) 

  return Im

def noMaxSupresion(Eo, Em):

  angles = Eo * 180. / np.pi
  angles[angles < 0] += 180
  
  (EmX, EmY) = Em.shape
  In = np.zeros((EmX,EmY))

  for x in range(1, EmX-1):
    for y in range(1, EmY-1):
      if Em[x,y] != 0:
        angle = angles[x,y]
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
          (a, b) = (Em[x, y+1], Em[x, y-1])
        elif (22.5 <= angle < 67.5):
          (a, b) = (Em[x+1, y-1], Em[x-1, y+1])
        elif (67.5 <= angle < 112.5):
          (a, b) = (Em[x+1, y], Em[x-1, y])
        elif (112.5 <= angle < 157.5):
          (a, b) = (Em[x-1, y-1], Em[x+1, y+1])

        if (Em[x,y] >= a) and (Em[x,y] >= b):
          In[x,y] = Em[x,y]
        else:
          In[x,y] = 0

  return In

def histeresis(strong, In):
  while strong:
    (i_x, i_y) = strong.pop()
    try:
      if In[i_x,i_y-1] == 0.5:
        In[i_x,i_y-1] = 1.0
        strong.append((i_x,i_y-1))

      if In[i_x,i_y+1] == 0.5:
        In[i_x,i_y+1] = 1.0
        strong.append((i_x,i_y+1))

      if In[i_x-1,i_y] == 0.5:
        In[i_x-1,i_y] = 1.0
        strong.append((i_x-1,i_y))

      if In[i_x+1,i_y] == 0.5:
        In[i_x+1,i_y] = 1.0
        strong.append((i_x+1,i_y))

      if In[i_x-1,i_y+1] == 0.5:
        In[i_x-1,i_y+1] = 1.0
        strong.append((i_x-1,i_y+1))

      if In[i_x-1,i_y-1] == 0.5:
        In[i_x-1,i_y-1] = 1.0
        strong.append((i_x-1,i_y-1))

      if In[i_x+1,i_y+1] == 0.5:
        In[i_x+1,i_y+1] = 1.0
        strong.append((i_x+1,i_y+1))

      if In[i_x+1,i_y-1] == 0.5:
        In[i_x+1,i_y-1] = 1.0
        strong.append((i_x+1,i_y-1))

    except IndexError as e:
      pass

  return np.where(In<1.0, 0.0, 1.0)


def edgeCanny(inImage, sigma, tlow, thigh):

  fImage = gaussianFilter(inImage, sigma)

  [Jx, Jy] = gradientImage(fImage, 'Sobel')

  Em = np.sqrt(np.power(Jx,2) + np.power(Jy,2))
  Eo = np.arctan2(Jy,Jx)

  In = noMaxSupresion(Eo, Em)

  (s_x, s_y) = np.where(In>thigh)
  (w_x, w_y) = np.where((In>=tlow )& (In<=thigh))
  (z_x, z_y) = np.where(In<tlow)

  In[(s_x, s_y)] = 1.0
  In[(w_x, w_y)] = 0.5
  In[(z_x, z_y)] = 0.0

  strong = list(zip(s_x.tolist(), s_y.tolist()))

  out = histeresis(strong, In)
  
  return out

def noMaxHarrys(R, radius, t):
  (rX, rY) = R.shape
  corners = np.zeros((rX, rY))
  skip = np.where(R > t, False, True) #Thresh hole

  #Avoid Downhill Beginning
  for i in range(radius, rX-radius):
    j = radius

    while j < rY-radius and (skip[i,j] or R[i,j-1] >= R[i,j]):
      j = j + 1
    
    while j < rY-radius:
      while j < rY-radius and (skip[i,j] or R[i, j+1]>= R[i,j]):
        j = j + 1

      if j < rY - radius:
        p1 = j + 2

        while p1 <= j + radius and R[i, p1] < R[i,j]:
          skip[i, p1] = True
          p1 = p1 + 1
        
        if p1 > j + radius:
          p2 = j-1

          while p2 >= j-radius and R[i,p2] <= R[i,j]:
            p2 = p2-1
          
          if p2 < j - radius:
            k = i + radius
            found = False 

            while not found and k > i :
              l = j + radius

              while not found and l>= j-radius:
                if R[k,l]>R[i,j]:
                  found = True
                else:
                  skip[k,l] = True
                l = l-1
              k = k-1
            k = i - radius

            while not found and k < i :
              l = j - radius
              while not found and l <= j+radius:
                if R[k,l] >= R[i,j]:
                  found = True
                l = l + 1 
              k = k + 1

            if not found:
              corners[i,j] = R[i,j]

        j=p1

  return corners

def cornerHarrys2(inImage, sigmaD, sigmaI, t):
  # k tipicamente entre [0.04,0.06]
  k = 0.04

  imageD = gaussianFilter(inImage, sigmaD)

  [Gx, Gy] = gradientImage(imageD, 'Sobel')

  A = Gx**2
  B = Gx*Gy
  C = Gy**2

  A_p = gaussianFilter(A, sigmaI) #XX
  B_p = gaussianFilter(B, sigmaI) #XY
  C_p = gaussianFilter(C, sigmaI) #YY

  det = (A_p * C_p) - (B_p**2)
  trace = A_p + C_p

  Rth =  det - (k * (trace ** 2))

  radius = int(np.ceil(3*sigmaI))
  return  noMaxHarrys(Rth, radius, t)

def cornerHarrys(inImage, sigmaD, sigmaI, t):
  # k tipicamente entre [0.04,0.06]
  k = 0.04

  imageD = gaussianFilter(inImage, sigmaD)

  [Gx, Gy] = gradientImage(imageD, 'Sobel')

  A = Gx**2
  B = Gx*Gy
  C = Gy**2

  A_p = gaussianFilter(A, sigmaI) #XX
  B_p = gaussianFilter(B, sigmaI) #XY
  C_p = gaussianFilter(C, sigmaI) #YY

  radius = int(np.ceil(3*sigmaI))
  (iX, iY) = inImage.shape
  Rth = np.zeros((iX, iY))

  for x in range(radius, iX - radius):
    for y in range(radius, iY - radius):
      Wxx = A_p[x-radius:x+radius+1, y-radius:y+radius+1]
      Wxy = B_p[x-radius:x+radius+1, y-radius:y+radius+1]
      Wyy = C_p[x-radius:x+radius+1, y-radius:y+radius+1]

      Sxx = Wxx.sum()
      Sxy = Wxy.sum()
      Syy = Wyy.sum()

      det = (Sxx * Syy) - (Sxy**2)
      trace = Sxx + Syy

      Rth[x,y] =  det - (k * (trace ** 2))

  
  return  [Rth,noMaxHarrys(Rth, radius, t)]