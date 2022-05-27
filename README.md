# Recognition System Based On Zernike Moments

<h2> detailed Conception </h2>
  <b> ZernikeVectors.py Contains the ZernikeVectorGenerator Class </b> : It has for attributes TestDir, ImageDir, ImageName. It has for methods GetImageName which returns the name of the image, BaseNET_segmentation which segments a binarized image, ScaleAndTranslationNormalization which normalizes a binarized image by scale and translation and TestGenerateZernikeVectors which takes the image provided by the user, calls the methods methods to start its processing (binarization, segmentation, normalization and calculation of and calculation of Zernike moments). The Zernike moment of the image provided by the the user will be compared to the moments of the images which are stored in a JSON FILE. This class will mobilize methods found in other classes.

  <hr>

 <b> ZernikeMomnets.py contains ZernikeMoments Class </b> Its only method is Descriptor. This method takes a processed image from the TestGenerateZernikeVector method of the class ZernikeVectorGenerator and returns the calculated Zernike moment.
  
 <hr>

