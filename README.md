# Probabilistic Image Hiding

## Requirements
* python 3.7

* pytorch 1.1.0

* ``pip install -r requirements.txt``

## Usage

### Training
* To train one model using one secret-cover image pair: 
```bash
python train_model.py
```

* To training 200 models using 200 secret-cover image pairs: 
```bash
python train_multiple_models.py
```

* The trained models will be saved in ``./TrainedModels`` directory.

### Evaluation
* An example of trained model is in ``./test-model`` directory.

* To evaluate the trained model by random sampling: 
```bash
python evaluate_model_random_sampling.py
```

* To evaluate the trained model by extracting secret image: 
```bash
python evaluate_model_secret_extraction.py
```

* The randomly sampled images and the extracted secret image will be saved in ``./test-model/Evaluation`` directory.

* The codes for calculating PSNR, SSIM, DISTS, SIFID, DS and KLD are provided in ``./Evaluation`` directory.

### Hiding Multiple Images
* To hide 2 secret images: 
```bash
python ./HidingMultipleImages/Hiding2Images/train_hiding2images.py
```

* To hide 3 secret images: 
```bash
python ./HidingMultipleImages/Hiding3Images/train_hiding3images.py
```

* To hide 4 secret images: 
```bash
python ./HidingMultipleImages/Hiding4Images/train_hiding4images.py
```

* The resulting models will be saved in ``./HidingMultipleImages/HidingxImages/TrainedModels`` directory, where "x" is the number of hidden secret images.

### Reference SinGAN without Hiding Image
* To train reference SinGAN without hiding any secret image, run: ``python ./NotHidingImage/train_original.py``

* The trained reference models will be saved in ``./NotHidingImage/TrainedModels`` directory.
