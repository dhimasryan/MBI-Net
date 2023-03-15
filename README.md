# MBI-Net: A Non-Intrusive Multi-Branched Speech Intelligibility Prediction Model for Hearing Aids

### Introduction ###

Improving the user's hearing ability to understand speech in noisy environments is critical to the development of hearing aid (HA) devices. For this, it is important to derive a metric that can fairly predict speech intelligibility for HA users. A straightforward approach is to conduct a subjective listening test and use the test results as an evaluation metric. However, conducting large-scale listening tests is time-consuming and expensive. Therefore, several evaluation metrics were derived as surrogates for subjective listening test results. In this study, we propose a multi-branched speech intelligibility prediction model (MBI-Net), for predicting the subjective intelligibility scores of HA users. MBI-Net consists of two branches of models, with each branch consisting of a hearing loss model, a cross-domain feature extraction module, and a speech intelligibility prediction model, to process speech signals from one channel. The outputs of the two branches are fused through a linear layer to obtain predicted speech intelligibility scores. Experimental results confirm the effectiveness of MBI-Net, which produces higher prediction scores than the baseline system in Track 1 and Track 2 on the Clarity Prediction Challenge 2022 dataset. 

For more detail please check our <a href="https://www.isca-speech.org/archive/pdfs/interspeech_2022/edozezario22_interspeech.pdf" target="_blank">Paper</a>

### Installation ###

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```

Please be noted, that the above environment is specifically used to run ```MBI-Net.py```. To generateSelf Supervised Learning (SSL) feature, please use ```python 3.6``` and follow the instructions in following <a href="https://github.com/microsoft/unilm/tree/master/wavlm" target="_blank">link</a>.  

### Extact SSL Feature ###

To extract the SSL feature, please use the following code:
```js
python Extract_SSL.py
```

### Train and Testing MBI-Net ###

Please use following script to train the model:
```js
python MBI-Net.py --gpus <assigned GPU> --mode train
```
For, the testing stage, plase use the following script:
```js
python MBI-Net.py --gpus <assigned GPU> --mode test
```

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
Zezario, R.E., Chen, F., Fuh, C.-S., Wang, H.-M., Tsao, Y. (2022) MBI-Net: A Non-Intrusive Multi-Branched Speech Intelligibility Prediction Model for Hearing Aids. Proc. Interspeech 2022, 3944-3948, doi: 10.21437/Interspeech.2022-10838

### Acknowledgement ###
We are grateful that our system received the best non-intrusive system at <a href="https://claritychallenge.org/clarity2022-workshop/results.html" target="_blank">Clarity Prediction Challenge 2022</a> 

### Note ###

<a href="https://github.com/CyberZHG/keras-self-attention" target="_blank">Self Attention</a>, <a href="https://github.com/grausof/keras-sincnet" target="_blank">SincNet</a>, <a href="https://github.com/microsoft/unilm/tree/master/wavlm" target="_blank">wavLM</a> are created by others
