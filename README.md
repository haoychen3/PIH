# Probabilistic Image Hiding

### Installation
1. python 3.7

2. pytorch 1.1.0

3. pip install -r requirements.txt

### Model Training
1. Training one model using one secret-cover image pair, run: ``python train_model.py``

2. Training 200 models using 200 secret-cover image pairs, run: ``python train_multiple_models.py``

3. The trained models will be saved in ``./TrainedModels`` directory.

### Model Evaluation
1. An example of trained model is in ``./test-model`` directory.

2. To evaluate the trained model by random sampling, run: ``python evaluate_model_random_sampling.py``

3. To evaluate the trained model by extracting secret image, run: ``python evaluate_model_secret_extraction.py``

4. The randomly sampled images and the extracted secret image will be saved in ``./test-model/Evaluation`` directory.

5. The codes for calculating PSNR, SSIM, DISTS, SIFID, DS and KLD are provided in ``./Evaluation`` directory.

### Hiding Multiple Images
1. To hide 2 secret images, run: python ./HidingMultipleImages/Hiding2Images/train_hiding2images.py
2. To hide 3 secret images, run: python ./HidingMultipleImages/Hiding3Images/train_hiding3images.py
3. To hide 4 secret images, run: python ./HidingMultipleImages/Hiding4Images/train_hiding4images.py
4. The resulting models will be saved in "./HidingMultipleImages/HidingxImages/TrainedModels" directory, where "x" is the number of hidden secret images.

### Reference SinGAN without Hiding Image
1. To train reference SinGAN without hiding any secret image, run: python ./NotHidingImage/train_original.py
2. The trained models will be saved in "./NotHidingImage/TrainedModels" directory.
