# DeepPavlov ODQA

### Introduction
I originally found, https://blog.tensorflow.org/2019/09/deeppavlov-open-source-library-for-end.html, when looking to find a way into Natural Language Processing (NLP) since it appears a mechanism for Articial General Intelligence (AGI).  Google uses BERT (Bidirectional Encoder Representations from Transformers) for search query interpretation and open sourced it in Nov 2018, https://blog.research.google/2018/11/open-sourcing-bert-state-of-art-pre.html.

DeepPavlov Open Domain Q&A (ODQA) uses:
* PyTorch (rather than TensorFlow)
* TF-IDF (Term Freq-Inverse Document Freq) for an index into the 14GB Wikipedia download DB
* SpaCy NLP
* Stanford Q&A Dataset (SQuAD)/ BERT Transformer (https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))

### Codebase obfuscation
* Needlessly loading TensorFlow

  Changed in 'site-packages/thinc/compat.py' (a requirement of SpaCy) with :
```
try:  # pragma: no cover
  # 19/12/23 DH: Prevent DeepPavlov loading TensorFlow (via Spacy) which it doesn't use
  print("*** thinc::compat.py - Not loading TensorFlow ***")
  raise ImportError

  import tensorflow
  import tensorflow.experimental.dlpack

  has_tensorflow = True
  has_tensorflow_gpu = len(tensorflow.config.get_visible_devices("GPU")) > 0
except ImportError:  # pragma: no cover
  tensorflow = None
  has_tensorflow = False
  has_tensorflow_gpu = False
```
  and 'TorchSquadTransformersPreprocessor' with :
```
$ time USE_TORCH=1 python -m deeppavlov interact en_odqa_infer_wiki
```

* Loading 14GB DB primary keys into 4GB of memory

  This job is done by TF-IDF

* Needlessly loading BPR (Binary Passage Retriever, https://arxiv.org/abs/2106.00882) Component into the Pipe

This changed the runtime for a basic question from about 6 mins on Colab to about 25 secs on my MacBook Pro with SSD.

### Patch for codebase
Take a copy of changed files (currently marked with "DH:" comments) in the codebase :
```
$ copy-deeppavlov-files
```
Update files in the codebase from a local copy :

```
$ update-deeppavlov-files
```

### ODQA Pipe
```
<JSON: en_odqa_infer_wiki.json>
  "in": ["question_raw"],

    Chainer[TfidfRanker]

      <JSON: en_ranker_tfidf_wiki.json>
        "in": ["docs"], ______________________
                       /                      |
          TfidfRanker /                       |
                     /       OBFUSCATION      |
    WikiSQLiteVocab /          CUTOUT         |
    StringMultiplier                          |
    LogitRanker   / (just press perforations) |
                 /____________________________|
      <JSON: qa_nq_psgcls_bert.json>
        "in": ["context_raw", "question_raw"],

          TorchSquadTransformersPreprocessor
          SquadBertMappingPreprocessor
          TorchTransformersSquad
          SquadBertAnsPostprocessor
```


