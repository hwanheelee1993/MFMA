# MFMA


## 1. Usage of pre-trained factual consistency checking model

MFMA is a pre-trained factual consistency checking model(trained as a binary classifier) for abstractive summaries trained with the augmented negative samples using Mask-and-Fill with Article(MFMA).
You only need huggingface transformers library to load the pre-trained model.

```
from transformers import AutoModelforSequenceClassification
model = AutoModelforSequenceClassification("mfma")
```

## 2. Training MFMA Instructions

<h2> Usage </h2>

<h3> 1) Install Prerequisites </h3>

Create a python 3.8 environment and then install the requirements.

Install packages using "requirements.txt"

```
conda create -name msm python=3.8
pip install -r requirements.txt
```

<h3> 2) Training MFMA </h3>

```
python train_fb.py --mask_ratio1 $MASK_ARTICLE \
                   --mask_ratio2 $MASK_ARTICLE \
```

<h3> 3) Generating Negative Summaries with MFMA </h3>

```
python infer_fb.py --mask_ratio1 $MASK_ARTICLE \
                   --mask_ratio2 $MASK_ARTICLE \
```

<h3> 4) Training Factual Consistency Checking Model using the Data </h3>

```
python train_metric.py --datadir $DATA_PATH \
```


```
@inproceedings{lee2022mfma,
      title={Masked Summarization to Generate Factually Inconsistent Summaries for Improved Factual Consistency Checking}, 
      author={Hwanhee Lee and Kang Min Yoo and Joonsuk Park and Hwaran Lee and Kyomin Jung},
      year={2022},
      month={july},
      booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
}
```
