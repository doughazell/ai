# Huggin API

* [Introduction](#introduction)
* [Scripts](#scripts)
  * [get-training-output](#get-training-output)
  * [get-model-output](#get-model-output)
  * [squad-interface.py](#squad-interfacepy)
  * [legend.py](#legendpy)
* [HuggingFace Transformers](#huggingface-transformers)
* [LIME process](#lime-process)
  * [Running 'lime.py'](#running-limepy)
  * [Running 'inDeCeptionV3.py'](#running-indeceptionv3py)

## Introduction
This is my Repo to learn about Transformer Neural Networks which has become a https://github.com/huggingface/transformers/tree/main/src/transformers API.

This has been a major journey for me which started by learning about Keras Sequential networks for https://en.wikipedia.org/wiki/MNIST_database processing with https://github.com/doughazell/mnist-training-errors.  Then moved onto using https://nbviewer.org/url/arteagac.github.io/blog/lime_image.ipynb with https://github.com/doughazell/ai/blob/main/lime.py (diagram below of LIME process).

## Scripts
* get-training-output
* get-model-output
* squad-interface.py
* legend.py

### get-training-output
This is a BASH script that makes trying things (like fine-tuning from Non-Pretrained) easy and includes:
* run_qa.py 

  ('run_qa.py' filename preserved from https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering#fine-tuning-bert-on-squad10)

  * Non-Pretrained via '"pretrained_model": true' in cfg.json specified in 'run_qa.py' execution

  * Ctrl-C in 'checkpointing.py::signal_handler(...)' allows fine-tuning SQUAD overnight (which took 2 weeks on my MacBook Pro with Flash Storage for 2 epochs)

  * It uses a 'trainer_qa.py::QuestionAnsweringTrainer'

    ```
    NOTE: 'run_qa.py' maps SQUAD to format required by Trainer ie:

    ----------------------------------------------------------------------------------------------------------------
    PRE: 'prepare_train_features': Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })

    POST: 'prepare_train_features': Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'],
        num_rows: 88524
    })
    ----------------------------------------------------------------------------------------------------------------

    Therefore 2 epochs of SQUAD results in a total training steps: 14754 (ie 88524 * 2 / 12)
      Number of epochs: 2
      Training samples: 88524
      Training batch size: 12
    ```

  Hooking 'modeling_bert.py' to collect logits + node weights during optimization via 'huggin_utils.py' :
  * BertSelfAttention.forward()

    ```
    ...
    def logSelectedNodeLogits(outputs):
      ...
      # 10/6/24 DH: Now sending 'tokenNum' since in TRAINING it is ALWAYS 384 but in NON-training it is TOKEN LENGTH
      huggin_utils.logSelectedNodeLogits(nodeForeachLogit, BertSelfAttention.cnt, bertLayerName="self", embedTokens=tokenNum)

    # Take FIRST BertSelfAttention + Node 287 (ie Max change) for all 384 logits
    if (BertSelfAttention.cnt == 1):
      logSelectedNodeLogits(outputs)

    # Take LAST BertSelfAttention + Node 287 (ie Max change) for all 384 logits
    if (BertSelfAttention.cnt == self.config.num_hidden_layers):
      logSelectedNodeLogits(outputs)
    ```

  * BertOutput.forward()
    ```
    ...
    if (BertSelfAttention.cnt == self.config.num_hidden_layers):
      huggin_utils.logSelectedNodeLogits(nodeForeachLogit, BertSelfAttention.cnt, bertLayerName="out", embedTokens=logitNum)

      # Reset counter for next training/non-training run NOW NOT DONE IN 'BertSelfAttention'
      BertSelfAttention.cnt = 0
    ```
  
  * BertForQuestionAnswering.forward()
    ```
    ...
    if logitsLen > 1: # ie training not non-training calc
      huggin_utils.logWeightings(self.qa_outputs.weight)

    if start_positions is not None and end_positions is not None:
      ...
      huggin_utils.logLogits(tokenizer, input_ids, start_logits, end_logits, start_loss, end_loss)
    ```

* graph-weights.py
* graph-losses.py
* graph-logits.py
* create-gv-training.py

in order to produce a PDF from Custom JSON data (which retrains several times with the same sample so a valid training comparison can be drawn) like:

![alt text](https://github.com/doughazell/ai/blob/main/huggingface/qa-training-10Aug.jpeg?raw=true)

or from SQUAD data (with 100,000+ samples so no point tracking same sample) like:

![alt text](https://github.com/doughazell/ai/blob/main/huggingface/qa-training-15Aug.jpeg?raw=true)

### get-model-output
* test-qa-efficacy.py

  Keep re-running until an answer is returned that is correct (to test partial fine-tuning after an overnight run of about 1000 epochs of batches of 12)

  Then add results to SQLite DB (currently '~/ai/bert/stack_trace.db')

  ```
  This DB was initially used for 'track-trainer.py' in order to get a stack trace at random time intervals in order to learn the HuggingFace training system.

  'track-trainer.py' used:
    'track-trainer-backend.py' (for creating a separate 'run_qa.py' process to signal Ctrl-C)
    'stop_trainer.py'
    'sort_error_log.py'
    'stop_trainer_utils.py'
  ```

  * db_utils.py

    Populate 'stack_trace.db::sample_indices'

    ```
    bert$ sqlite3 stack_trace.db

    sqlite> select * from sample_indices;
    id|model_efficacy_id|seq_num|seq_ids
    ...
    15|8|36|73046,28615,43280,35492,56949,5569,18151,40362,66763,47757,85209,637,66349,78306,44359,41921,85703,3223,58179,48952,54222,80427,19808,40113,8465,54659,61244,32650,75646,17769,61959,63990,79124,24554,59255,4916
    ```

  * db-indices.py

    Populate 'stack_trace.db::model_efficacy' with each record being a distinct "model/training state/training data"

    ```
    bert$ sqlite3 stack_trace.db

    sqlite> select * from model_efficacy;
    id|model_type_state|correct_num|sample_num|sample_seq
    1|BertForQuestionAnswering-1026-squad|17|51|3,3,6
    2|BertForQuestionAnswering-1026-data.json|7|7|
    3|BertForQuestionAnswering-40-data.json|2|2|
    4|BertForQuestionAnswering-1126-data.json|1|1|
    5|BertForQuestionAnswering-NoPretrain-1126-squad|15|1584|216,60,27,354,69,162,33,150,156
    6|BertForQuestionAnswering-10-data.json|3|3|1,1,1
    7|BertForQuestionAnswering-NoPretrain-2134-squad|17|1152|159,78,36,15,66,147,72,3,87,3,42,30,42,12,48,102,210
    8|BertForQuestionAnswering-NoPretrain-3188-squad|13|354|81,6,18,63,3,48,57,9,3,3,27,36
    ```

    so you can see from "id 8" that "BertForQuestionAnswering-NoPretrain-3188-squad" is 13/354 (3.7%) accurate and the last run of 'test-qa-efficacy.py' took 36 attempts to get a correct answer with the SQUAD ids listed in "id 15" of 'sample_indices'.

* graph-weights.py
* graph-losses.py
* graph-node-logits.py
* create-gv-output.py

![alt text](https://github.com/doughazell/ai/blob/main/huggingface/qa-output-16Aug.jpeg?raw=true)

### squad-interface.py
There didn't seem to be an online DB to get the Context/Question/Correct Answer for SQUAD index (https://huggingface.co/datasets/rajpurkar/squad didn't provide a mechanism to search on index).  This will provide an indication of the grammar level of a "model/training state" by human inference of failures/successes.

```
huggingface$ python squad-interface.py
  87599 samples + 10570 validation samples = 98169 total in 'squad'

  Do you want to check for duplicate 'seq_ids'? [Y/n] 

  Opened connection to DB:  /Users/doug/src/ai/bert/stack_trace.db

  CHECKING FOR DUPLICATE 'seq_ids'
  --------------------------------
  RECORD NUMBER: 15

  SET: 7 - startIdx: 0, endIdx: 2
    ...
    Ids in record 1 & 2: '[64745]'
    Ids in record 2 & 0: ''
    Ids in record 2 & 1: '[64745]'

  TOGGLING TO SET: 8 at index 3
  SET: 8 - startIdx: 3, endIdx: 14
    ...
    Ids in record 6 & 7: '[23581, 21533, 46079]'
    ...
    Ids in record 6 & 12: '[23581, 21533, 46079]'
    ...
    Ids in record 7 & 6: '[23581, 21533, 46079]'
    ...
    Ids in record 7 & 12: '[23581, 21533, 46079]'
    ...
    Ids in record 12 & 6: '[23581, 21533, 46079]'
    Ids in record 12 & 7: '[23581, 21533, 46079]'
    ...

  What index do you want? (just press return for random index) 
  Getting 'random' index: 17363

  INDEX: 17363 (of 87599)
  QUESTION: Who was the member of Parliament who brought a bill about Daylight Saving Time to the House of Commons in 1908?
  CONTEXT: Modern DST was first proposed by the New Zealand entomologist George Hudson, whose shift-work job gave him leisure time to collect insects, and led him to value after-hours daylight. In 1895 he presented a paper to the Wellington Philosophical Society proposing a two-hour daylight-saving shift, and after considerable interest was expressed in Christchurch, he followed up in an 1898 paper. Many publications credit DST's proposal to the prominent English builder and outdoorsman William Willett, who independently conceived DST in 1905 during a pre-breakfast ride, when he observed with dismay how many Londoners slept through a large part of a summer's day. An avid golfer, he also disliked cutting short his round at dusk. His solution was to advance the clock during the summer months, a proposal he published two years later. The proposal was taken up by the Liberal Member of Parliament (MP) Robert Pearce, who introduced the first Daylight Saving Bill to the House of Commons on 12 February 1908. A select committee was set up to examine the issue, but Pearce's bill did not become law, and several other bills failed in the following years. Willett lobbied for the proposal in the UK until his death in 1915.
  EXPECTED ANSWER: Robert Pearce
  EXPECTED ANSWER START IDX: 899
  TOKEN LEN (Bert vocab): 271

```

### legend.py
Written to add a legend to the generated GraphViz graph from 'torchview'

(See comments in file for details)

## HuggingFace Transformers
* Activation/Optimization
* Logits via 'hidden_states'

# LIME process
LIME (https://arxiv.org/abs/1602.04938) works by having a Binomial Distribution of image masking (ie removing segments of the full image) in order to perturb the image prior to calling 'keras.applications.inception_v3.InceptionV3().predict(perturbed_img)'.  Then get the RMS segment diff from the orig image to each of Binomially Distributed segment masks.  Finally correlate the RMS diff with the place of the predicted full image.  This then provides an order to segment importance of the final prediction (to compare how you would ID the same image and therefore gain confidence in the prediction).

A good place to start with understanding LIME is:
```
'lime.py::runLimeAnalysis()'

  # Step 1/4
  limeImage.createRandomPertubations()

  # Step 2/4
  limeImage.getPredictions()
  
  # Step 3/4
  limeImage.getDistanceWeights()
  
  # Step 4/4
  limeImage.getLinearRegressionCoefficients()

    -------------------------------------
    Final step (4/4) in: getLinearRegressionCoefficients
                         -------------------------------
     to correlate InceptionV3 prediction of full image (class index 208 )
     with masked image predictions, inceptionV3_model.predict()  (in Step 2/4, 'Lime.getPredictions()')
     using weights obtained for mask distance from full image    (in Step 3/4, 'Lime.getDistanceWeights()')

    class_to_explain: 208 , from all InceptionV3 trained classes: (100, 1, 1000)
    LinearRegression.fit(): mask perturbations: (100, 28) , prediction: (100, 1)
     eg...Xvals[0]: [1 1 1 1 0 0 1 0 1 0 1 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 1] , yVals[0]: [0.07313019]

    Multiple LinearRegression() coeffs (from weights): (28,)
     eg...lowest: -0.10693367968250382 , highest: 0.3781445941220827
     (100 pertubation masks for 28 segments leads to a linear correlation line of importance of each segment in full image)
    -------------------------------------
```

## Diagram overview
* Top Row: Get mask from Binomial Distribution of which of 28 segments to include

  eg Top prediction of first perturbation: [1 1 1 1 0 0 1 0 1 0 1 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 1] = conch 
  
* Row 2: Get prediction place for 'Labrador_retriever' (ie top prediction for full image) + RMS diff between segment mask used and full image mask

  eg [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
  
* Row 3: Perform a multiple linear regression for prediction place of 'Labrador_retriever' vs 28 bit segment mask
* Bottom Row: 'InceptionV3().predict()' produces order of 1000 known images (eg 'Labrador_retriever', 'conch' etc) and need to correlate placing of image 208 (ie 'Labrador_retriever') for each of 100 * 28 bit masks.

![alt text](https://github.com/doughazell/ai/blob/main/LIME-flow-diag.jpg?raw=true)

## Running 'lime.py'
```
$ python lime.py
```
Gives:

![alt text](https://github.com/doughazell/ai/blob/main/segment_coeffs.png?raw=true)

and

![alt text](https://github.com/doughazell/ai/blob/main/top_segments.png?raw=true)

## Running 'inDeCeptionV3.py'
This is initially a re-run of 'lime.py' but then blanks out the top 4 segments to identity the image (which is now a cat).

```
$ python inDeCeptionV3.py

  Predictions summary
  -------------------
  (Run: 'open https://observablehq.com/@mbostock/imagenet-hierarchy')

  Labrador_retriever :
    ('n02099712', 'Labrador_retriever', 0.818291)
    ('n02099601', 'golden_retriever', 0.015929893)
    ('n02093428', 'American_Staffordshire_terrier', 0.010066423)

  Egyptian_cat :
    ('n02124075', 'Egyptian_cat', 0.098595686)
    ('n02108915', 'French_bulldog', 0.09533588)
    ('n01883070', 'wombat', 0.08558813)

  NOTE THE PLACING + PROBABILITY of the second prediction in the full image (ie first prediction): 
  ==============================

    Labrador_retriever entry 18 : ['n02124075' 'Egyptian_cat' '0.0018015162']

  ----------------------------------------------------------------
```
Gives on the first run:

![alt text](https://github.com/doughazell/ai/blob/main/top_features_12.png?raw=true)

and then on the second run (having CLOSED ALL the previous windows)

![alt text](https://github.com/doughazell/ai/blob/main/top_features_19.png?raw=true)