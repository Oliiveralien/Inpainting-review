## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
```

## Evaluation

### Download the [model](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view?usp=sharing) and place it in ./pretrained_models/

#### Testing on GoPro dataset
- Download [images](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf?usp=sharing) of GoPro and place them in `./Datasets/GoPro/test/`
- Run
```
python test.py --dataset GoPro
```

#### Testing on HIDE dataset
- Download [images](https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK?usp=sharing) of HIDE and place them in `./Datasets/HIDE/test/`
- Run
```
python test.py --dataset HIDE
```


#### Testing on RealBlur-J dataset
- Download [images](https://drive.google.com/drive/folders/1KYtzeKCiDRX9DSvC-upHrCqvC4sPAiJ1?usp=sharing) of RealBlur-J and place them in `./Datasets/RealBlur_J/test/`
- Run
```
python test.py --dataset RealBlur_J
```



#### Testing on RealBlur-R dataset
- Download [images](https://drive.google.com/drive/folders/1EwDoajf5nStPIAcU4s9rdc8SPzfm3tW1?usp=sharing) of RealBlur-R and place them in `./Datasets/RealBlur_R/test/`
- Run
```
python test.py --dataset RealBlur_R
```

#### To reproduce PSNR/SSIM scores of the paper on GoPro and HIDE datasets, run this MATLAB script
```
evaluate_GOPRO_HIDE.m 
```

#### To reproduce PSNR/SSIM scores of the paper on RealBlur dataset, run
```
evaluate_RealBlur.m 
```
