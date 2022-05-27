# Recognition System Based On Zernike Moments

<h2> detailed Conception </h2>
  <b> ZernikeVectors.py Contains the ZernikeVectorGenerator Class </b> : It has for attributes TestDir, ImageDir, ImageName. It has for methods GetImageName which returns the name of the image, BaseNET_segmentation which segments a binarized image, ScaleAndTranslationNormalization which normalizes a binarized image by scale and translation and TestGenerateZernikeVectors which takes the image provided by the user, calls the methods methods to start its processing (binarization, segmentation, normalization and calculation of and calculation of Zernike moments). The Zernike moment of the image provided by the the user will be compared to the moments of the images which are stored in a JSON FILE. This class will mobilize methods found in other classes.

<br>
  

 <b> ZernikeMomnets.py contains ZernikeMoments Class </b>: Its only method is Descriptor. This method takes a processed image from the TestGenerateZernikeVector method of the class ZernikeVectorGenerator and returns the calculated Zernike moment.
  

 
 <b>OptimalSimilartyMeasure.py contains OptimalSimilarityMeasure Class </b> :  which calculates the optimal measure based on the (See Chapter 3 section 3.10.3 and 3.10.4 in the attached Rapport) using the methods ComputeC, ComputeAq, ComputeBq, ComputeAq_Bq_over_q, Overall_func_d, Overall_func_deriv_d
 

 
 
 <b> Searcher.py contains  SearcherOptimal Class </b> : It has as attribute the JSON file which contains the moments
of all the stored images. It has a single Search method which
takes as input the Zernike vector of the image given by the user, browses the whole JSON file and computes the difference (score) between the moment of the image provided by the user with that of each image. Then it will create a list of tuples: image name + its score. An ascending sort according to the value of the score will be done.




<b> test_images_for_zernike/ contains images </b> : images to test the system 



<b>  static/DataBase conatins json file </b> : this json file conatins a map name of status : calculated zernike vector { statuName1 : vector1 , ....

<b>  static/bardo_pcd_pics conatins images </b> : images available in our DataBase




