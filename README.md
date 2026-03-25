# Assignment 4: Automated track infrastructure recognition using vibration analysis

Tor Suneson

D7015B-IAI-and-eM

## *Main files*

**Code 1-1.ipynb** Completed the original code.

*Requires*

kaleido ( pip install -U kaleido )

How to run: Run Jypiter code block

**Code2.ipynb** Comlete flow for selecting files and train classic models and Deepearning

```text
1) 

import importlib, select_runs
importlib.reload(select_runs)
manifest = select_runs.build_manifest()
```

```text
# 2) saves data and maps in Code2_exports for reviewing 
# Not a nessesary step to train models since the exoprt dir is not used for anything. 


%run Code2.py
```

```text
# 3) 
# train Classic models and Deep learning model Keras MLP
# 
# run select_runs to get manifest


# Deep learning orchestration
RUN_DEEPLEARNING = False  # run MLP at the same time as Classic models 
MLP_ONLY = False     # true will activate MLPClassification only , without weighted_class
RUN_KERAS_WEIGHTED = True   # To train and save Keras model with weighted_class, at the same time as classic models
KERAS_PREDICT_ONLY = False  # do not run prediction from stored model. 
# 
# 

%run label_and_train.py #<- new 
```

```text
# 4) 
# plot data 1 and refined data in same map to see how well the data got refined visually 
# last step 


%run Plot_as_Code1.py
```

**KerasPoc.ipynb**nb Complete flow for predicting fromtr trained model

*Requires ( saved after Code2 run above)*

./mlp/keras_mlp_weighted.keras

./mlp/keras_scaler.npz

```text
# 1) 
# Build quick_manifest.json manually from runs out to falun or gävle

# Prebuild alternatives 
#  quick_manifest Borlänge - Mjölby.json
#  quick_manifest Mora - Borlänge.json
#  
# rename either to 
#  quick_manifest.json 
```

```text
# 2) 
# refine an build selected_runs

import importlib, select_runs as sr
importlib.reload(sr)
#manifest = sr.build_manifest()
#quick_manifest = sr.build_quick_manifest()

refined = sr.build_selected_runs_from_quick_manifest_no_thresholds(
    quick_manifest_path="quick_manifest.json",
    out_manifest_path="selected_runs.json",
    overwrite_refine=False
)
```

```text
# 3) -ish 
# This step is introduced in next step Label and Train IF segments_inference.csv does not exist
# 
import importlib, label as lab
importlib.reload(lab)
from pathlib import Path


df_inf = lab.build_inference_dataset(Path("selected_runs.json"), Path("Data 2"))
df_inf.to_csv("segments_inference.csv", index=False)
print("Saved:", "segments_inference.csv", "rows=", len(df_inf))
```

```text
# 3) 
# For Keras Predict only , set parameters in label_and_train as follows. 

## Deep learning orchestration
RUN_DEEPLEARNING = False #True
MLP_ONLY = False
RUN_KERAS_WEIGHTED = False   # kör Keras MLP med class weights som komplement
KERAS_PREDICT_ONLY = True   # run prediction only, require that RUN_KERAS_WEIGHTED har been run once

# make sure you have trained model in advanded so that 
# .mlp/keras_mlp_weighted.keras
# .mlp/keras_scaler.npz
# are avaiable 

%run label_and_train.py
```

```text
# 4) Plot the result 
#
# .plots_map/map_all.html
# .plots_map/map_all.png


%run Plot_as_Code1_keras.py
```

## *Data set (not provided)*

./Data 1/

./Data 2/

## *Support files:*

### Code 2 breakout

[Code2.py](http://Code2.py)

### Selecting files

select_[runs.py](http://runs.py)

refine_data.py

### Label and Train

label_and_[train.py](http://train.py) (wrapper)

[label.py](http://label.py)

[train.py](http://train.py)

### Libraries for feature selection and deep learning

feature_[deeplearning.py](http://deeplearning.py)

feature_[embedded.py](http://embedded.py) - from Assignment 3

feature_[filters.py](http://filters.py) - from Assignment 3

feature_[wrappers.py](http://wrappers.py) - from Assignment 3

### Keras Deep learning

keras_mlp_[weighted.py](http://weighted.py)

ensemble_[combine.py](http://combine.py)

### Manifest files

selected_runs_spline.json

selected_runs_quick.json

selected_runs Borlänge Örebro.json

**Rename one of these to:&#x20;**&#x73;elected_runs.json

quick_manifest Borlänge - Mjölby.json

quick_manifest Mora - Borlänge.json

**Rename one of these to:** quick_manifest.json

### Plot files

Plot_as_[Code1.py](http://Code1.py)

Plot_as_Code1_[keras.py](http://keras.py)

### Debugfiles

run_scores_debug.csv
