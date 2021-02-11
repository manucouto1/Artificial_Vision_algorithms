import numpy as np
import matplotlib.pyplot as plt

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
  copiaB[cx:cx+ix, cy:cy+iy] = inImage

  for x in range(0, ix):
    for y in range(0, iy):
      if inImage[x,y]==0.0 :
        copiaB[x:x+p, y:y+q] = np.minimum((1.0-SE), copiaB[x:x+p, y:y+q])
        
  return copiaB[cx:cx+ix, cy:cy+iy]

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

  if not SE:SE = np.array([[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,0.0]])
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

def gradientImage(inImage, operador):
  op_types = ['Roberts', 'CentralDiff','Prewitt','Sobel']
  op_masks = {
      op_types[0] : (np.array([[-1,0],[0,1]]),np.array([[0,-1],[1,0]]), (0,0)),
      op_types[1] : (np.array([-1,0,1]),np.array([[-1,0,1]]).T, ()), ## Se divide entre 2?
      op_types[2] : (np.array([[-1,0,1],[-1,0,1],[-1,0,1]]), np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).T, ()),
      op_types[3] : (np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T, ()) ##Según teoría se puede optimizar separando la matrix
  }

  if operador not in op_types:
    raise Exception("Wrong op, ",operador, " not in ", op_types)
  
  (maskX, maskY, centers) = op_masks.get(operador)

  Gx = conV1(inImage, maskX, centers)
  Gy = conV1(inImage, maskY, centers)

  return [Gx, Gy]

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

def neightBours(angle, Em, x, y):
  if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
    return (Em[x, y+1], Em[x, y-1])
  elif (22.5 <= angle < 67.5):
    return (Em[x+1, y-1], Em[x-1, y+1])
  elif (67.5 <= angle < 112.5):
    return (Em[x+1, y], Em[x-1, y])
  elif (112.5 <= angle < 157.5):
    return (Em[x-1, y-1], Em[x+1, y+1])

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

  In = np.where(In<1.0, 0.0, 1.0)

  return In

def edgeCanny(inImage, sigma, tlow, thigh):

  fImage = gaussianFilter(inImage, sigma)

  [Jx, Jy] = gradientImage(fImage, 'Sobel')

  Em = np.sqrt(np.power(Jx,2) + np.power(Jy,2))
  Eo = np.arctan2(Jy,Jx)

  angle = Eo * 180. / np.pi
  angle[angle < 0] += 180
  
  (EmX, EmY) = Em.shape
  In = np.zeros((EmX,EmY))

  for x in range(1, EmX-1):
    for y in range(1, EmY-1):
      if Em[x,y] != 0:
        (a, b) = neightBours(angle[x][y],Em, x, y)

        if (Em[x,y] >= a) and (Em[x,y] >= b):
          In[x,y] = Em[x,y]
        else:
          In[x,y] = 0

  (s_x, s_y) = np.where(In>thigh)
  (w_x, w_y) = np.where((In>=tlow )& (In<=thigh))
  (z_x, z_y) = np.where(In<tlow)

  In[(s_x, s_y)] = 1.0
  In[(w_x, w_y)] = 0.5
  In[(z_x, z_y)] = 0.0

  strong = list(zip(s_x.tolist(), s_y.tolist()))

  out = histeresis(strong, In)
  
  return out

def noMaxHarrys(det, trac):
  (detX, detY) = det.shape
  minDet = np.max(det)*0.1
  minTrac = np.max(trac)*0.1

  for i in range(0, detX):
    for j in range(0, detY):
      if det[i,j] < minDet or trac[i,j] < minTrac:
        det[i,j] = 0.0
  
  return det

def CornerHarrys(inImage, sigmaD, sigmaI, t):
  k = 0.05
  w = 7

  (iX, iY) = inImage.shape
  (wOffst) = int(w/2)

  imageD = gaussianFilter(inImage, sigmaD)
  [Gx, Gy] = gradientImage(imageD, 'Sobel')

  GIx = gaussianFilter(Gx, sigmaI)
  GIy = gaussianFilter(Gy, sigmaI)

  xx = GIx**2
  xy = GIx*GIy
  yy = GIy**2

  corners = []

  for x in range(wOffst, iX-wOffst):
    for y in range(wOffst, iY-wOffst):
      sXX = xx[x-wOffst:x+wOffst+1, y-wOffst:y+wOffst+1].sum()
      sXY = xy[x-wOffst:x+wOffst+1, y-wOffst:y+wOffst+1].sum()
      sYY = yy[x-wOffst:x+wOffst+1, y-wOffst:y+wOffst+1].sum()

      det = (sXX * sYY) - (sXY ** 2)
      trace = sXX + sXY

      det = noMaxHarrys(det, trace)

      M = det - k * (trace ** 2)

      if(M[x,y] > t):
        corners.append((x,y))

  return corners

      
