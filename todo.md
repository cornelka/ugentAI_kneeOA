## general
- [OK] decide on augmentation steps
- decide on model for latent and gradcam
- cleanup code
- cleanup unneeded data
- [OK] next meeting(s)
- [OK] do we need more time ?

- make sure we can run all notebooks end2end repeatedly
- install instructions ?


## preprocessing.ipynb
- [OK] torchvision OK or obliged to use pillow
- [] recheck explanations (normalization) / versie in preprop is OK
- [OK] mean + std check met transferlearning + carlo  / versie in preprop is OK
- [OK] testdata: toepassing mean/std van train op test ? bereken op train  toepassen op test
- [OK] specific options needed for baseline /transfer ? / toegevoegd


##  baseline.ipynb -> OK in baseline2
- use online augmentation from preprocessing
- save model structure + weights for best model
- label the models structure  (we need this for notebook 4 + 5)
self.conv_features   # full conv stack
self.last_conv       # for Grad-CAM
self.embedding       # latent vector (128-d)
self.classifier      # final layer


##  transferlearning.ipynb
- use online augmentation from preprocessing  (same as baseline)
- save model structure + weights for best model
- label the models structure  (we need this for notebook 4 + 5) (via hooks)
self.conv_features   # full conv stack
self.last_conv       # for Grad-CAM
self.embedding       # latent vector (128-d)
self.classifier      # final layer

- assignment states to adapt data not the model ; (indicate model adaption alternative and why not used)

- dropout-rate (not same as baseline, is ok why)
- code is not running, lots of open sections


##  latentspace.ipynb
- [ok] open best model instead of code-copy; access latent layer
- rerun and addapt on final best model
- revisit code and add explanations
- choose neighbours visualisation
- clinical interpretations

##  GradCAM.ipynb
- [ok] open best model instead of code-copy; access cam layer
- rerun and addapt on final best model
- check the code and logic
- understand/explain it
- clinical interpretations (seems to learn on invalid regions at the moment)



##  report
- restructure preprocessing part + check quotes
- write other parts
