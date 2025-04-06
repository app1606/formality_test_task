# Improving Writing Assistance Test Task Solution

The notebook with the solution may be found here: `notebooks/Test_Task_Formality.ipynb`. This notebook contains code with my comments and conclusions. The work was done and ran in Colab, so I would recommend using this [link](https://colab.research.google.com/drive/1E6BI9-al-qR2__cuRsbXdzA8q0IeMZ7I?usp=sharing) to check the results yourself. In order to do so, one has to add files

```
data/formal_sentences.rtf
data/informal_sentences.rtf
data/middle_sentences.rtf
```
to the `/content` folder of Colab notebook. These files contain the sentences generated by ChatGPT. They are used to create a diverse dataset of formal and informal sentences. 

Folder `scripts` contains scripts for data creation, preparation and model evaluation. `notebooks/Test_Task_Formality_usage_examples.ipynb` [displays](https://colab.research.google.com/drive/1jyXCSTDfGzwtiLzvWNl2XGCFeidKF0U8?usp=sharing) how to run the scripts to reproduce the results without extra comments from my sides. To do so one has to run the notebook in Colab and add all files from the `scripts` folder to the `/content` folder in Colab Notebook.  

# Documentation

## Datasets

[SQUINKY!](https://arxiv.org/abs/1506.02306) dataset was used to evaluate the models along with the sentences generated by ChatGPT. 

## Libraries

Libraries and frameworks used:

- **Libraries**:
  - `pandas` — data manipulation 
  - `numpy 1.26.4` — numerical computations
  - `torch` — tensor computations
  - `scikit-learn` — metrics computation
  - `matplotlib` — visualization
  - `transformers` — pre-trained LLMs
- **Models**:
  - `deberta-large-formality-ranker` — pre-trained, from [HuggingFace](https://huggingface.co/s-nlp/deberta-large-formality-ranker)
  - `xlmr_formality_classifier ` — pre-trained, from [HuggingFace](https://huggingface.co/s-nlp/xlmr_formality_classifier)
  - `CatboostClassifier` — fine-tuned in this work, from [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)

## Metrics

- **Metrics Used**:
  - Accuracy
  - F1-score
  - Precision / Recall

These metrics are widely used in this field. Accuracy shows the part of correctly classified test samples. Precision for a class displays which part of positively-classified samples are actuall positive. Recall tells us which part of the positive samples were classified correctly. F1 is used as a balanced metric, the harmonic mean of precision and recall, that is sensitive to the inbalance between classes. Metrics were calculated using `scikit-learn` and displayed with `classification_report` method.

## Conclusions

I managed to show that fine-tuned DeBERTa model outperforms fine-tuned XLMR-Roberta model on all the selected metrics, although both of them lose to CatBoostClassifier. I believe that it happened because I have used samples from my own dataset to fine-tune Catboost model and then ran evaluation over the test part of this dataset. Since the model was trained on the data from the same source, it had advantage over others. 

## Results 

| Model    | Formal Precision | Formal Recall | Formal F1 |  Informal Precision | Informal Recall | Informal F1 |Accuracy |
| -------- | ---------|-| -| -|-| -| -|
| DeBERTa  | 0.8    | 0.77| 0.78| 0.76|0.79| 0.78| 0.78|
| XLMR-Roberta | 0.76   | 0.5| 0.6| 0.61|0.83| 0.7| 0.66|
| Catboost    | 0.82    | 0.83| 0.83| 0.83|0.82| 0.83| 0.83|
