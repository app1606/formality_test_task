# Improving Writing Assistance Test Task Solution

The notebook with the solution may be found in this folder, `Test_Task_Formality.ipynb`. The work was done and ran in Colab, so I would recommend using this [link](https://colab.research.google.com/drive/1E6BI9-al-qR2__cuRsbXdzA8q0IeMZ7I#scrollTo=nhsJQiaolwCj) to check the results yourself. In order to do so, one has to add files

```
formal_sentences.rtf
informal_sentences.rtf
middle_sentences.rtf
```
to the content folder. These files contain the sentences generated by ChatGPT. They are used to create a diverse dataset of formal and informal sentences. 

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

## Metrics

- **Metrics Used**:
  - Accuracy
  - F1-score
  - Precision / Recall

These metrics are widely used in this field. Accuracy shows the part of correctly classified test samples. Precision for a class displays which part of positively-classified samples are actuall positive. Recall tells us which part of the positive samples were classified correctly. Metrics were calculated using `scikit-learn` and displayed with `classification_report` method.


