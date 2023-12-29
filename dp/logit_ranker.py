# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from operator import itemgetter
from typing import List, Union, Tuple, Optional

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.doc_retrieval.utils import find_answer_sentence

logger = getLogger(__name__)


@register("logit_ranker")
class LogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.

     Args:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
        sort_noans: whether to downgrade noans tokens in the most possible answers
        top_n: number of answers to return

     Attributes:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
        top_n: number of answers to return

    """

    def __init__(self, squad_model: Union[Chainer, Component], batch_size: int = 50,
                 sort_noans: bool = False, top_n: int = 1, return_answer_sentence: bool = False, **kwargs):
        self.squad_model = squad_model
        self.batch_size = batch_size
        self.sort_noans = sort_noans
        self.top_n = top_n
        self.return_answer_sentence = return_answer_sentence

    def __call__(self, contexts_batch: List[List[str]], questions_batch: List[List[str]],
                 doc_ids_batch: Optional[List[List[str]]] = None) -> \
            Union[
                Tuple[List[str], List[float], List[int], List[str]],
                Tuple[List[List[str]], List[List[float]], List[List[int]], List[List[str]]],
                Tuple[List[str], List[float], List[int]],
                Tuple[List[List[str]], List[List[float]], List[List[int]]]
            ]:

        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.

        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config
            doc_ids_batch (optional): names of the documents from which the contexts_batch was derived
        Returns:
             a batch of best answers, their scores, places in contexts
             and doc_ids for this answers if doc_ids_batch were passed
        """
        if doc_ids_batch is None:
            logger.info("you didn't pass tfidf_doc_ids as input in logit_ranker config so "
                           "batch_best_answers_doc_ids can't be compute")

        batch_best_answers = []
        batch_best_answers_score = []
        batch_best_answers_place = []
        batch_best_answers_doc_ids = []
        batch_best_answers_sentences = []
        for quest_ind, [contexts, questions] in enumerate(zip(contexts_batch, questions_batch)):
            # 17/12/23 DH:
            print("...just slip a cheeky one in here, stop:",len(contexts), ", step:",self.batch_size)
            results = []
            for i in range(0, len(contexts), self.batch_size):
                c_batch = contexts[i: i + self.batch_size]
                q_batch = questions[i: i + self.batch_size]
                
                print("  calling squad_model with: ")
                print("    c_batch: ",len(c_batch))
                print("    q_batch: ",len(q_batch))
                # 26/12/23 DH:
                if i == 0:
                    print()
                    print("Context 1: ", c_batch[0])
                    print("Question 1: ", q_batch[0])
                
                batch_predict = list(zip(*self.squad_model(c_batch, q_batch), c_batch))
                results += batch_predict
            if self.sort_noans:
                results_sort = sorted(results, key=lambda x: (x[0] != '', x[2]), reverse=True)
            else:
                results_sort = sorted(results, key=itemgetter(2), reverse=True)

            # 22/12/23 DH: Change 'best_answers' to longest ngram found in 'HashingTfIdfVectorizer' (by 'StreamSpacyTokenizer')
            
            # 23/12/23 DH: Probably better to access via member cascade (rather than Class method of Class member)
            #              'TfidfRanker.vectorizer' -> 'HashingTfIdfVectorizer.tokenizer'
            print("Getting 'ngram' in 'LogitRanker' via 'StreamSpacyTokenizer._getLongestNGram()'")
            from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer

            ngram = StreamSpacyTokenizer._getLongestNGram()
            best_answers = [ngram for x in results_sort[:self.top_n]]

            #best_answers = [x[0] for x in results_sort[:self.top_n]]
            best_answers_place = [x[1] for x in results_sort[:self.top_n]]
            best_answers_score = [x[2] for x in results_sort[:self.top_n]]
            best_answers_contexts = [x[3] for x in results_sort[:self.top_n]]

            # 18/12/23 DH: Store to 'batch' (WHY ???)
            batch_best_answers.append(best_answers)
            batch_best_answers_place.append(best_answers_place)
            batch_best_answers_score.append(best_answers_score)
            
            # 27/12/23 DH: Changed order of 'doc_ids' & 'best_answers_sentences'

            if doc_ids_batch is not None:
                doc_ind = [results.index(x) for x in results_sort]
                
                # 27/12/23 DH: De-regex'ing :
                """
                #for i in doc_ind:
                #    print( doc_ids_batch[0][i] )
                
                "hello world"

                |2|4|1|3|0|_|2|4|1|3|0|
               
                 0 1 2 3 4 _ 0 1 2 3 4
                |o|l|h|l|e|_|d|r|w|l|o|
                """

                batch_best_answers_doc_ids.append(
                    [doc_ids_batch[quest_ind][i] for i in doc_ind] [:len(batch_best_answers[-1])]
                )
            
            best_answers_sentences = []
            for id, answer, place, context in zip(batch_best_answers_doc_ids[0], best_answers, best_answers_place, best_answers_contexts):
                sentence = find_answer_sentence(place, context)
                print()
                print("  adding: ",sentence)
                print("  from: ",context)
                print("  text index: ",place)
                print("  DB id: ",id)
                print()
                best_answers_sentences.append(sentence)
            batch_best_answers_sentences.append(best_answers_sentences)

        # END: --- for quest_ind, [contexts, questions] in enumerate(zip(contexts_batch, questions_batch)) ---

        if self.top_n == 1:
            print("top_n == 1")
            batch_best_answers = [x[0] for x in batch_best_answers]
            batch_best_answers_place = [x[0] for x in batch_best_answers_place]
            batch_best_answers_score = [x[0] for x in batch_best_answers_score]
            batch_best_answers_doc_ids = [x[0] for x in batch_best_answers_doc_ids]
            batch_best_answers_sentences = [x[0] for x in batch_best_answers_sentences]

        # 20/12/23 DH: https://docs.python.org/2.7/howto/doanddont.html#using-backslash-to-continue-statements
        #              so I have replaced '\' with '()' on return statements
        if doc_ids_batch is None:
            print("doc_ids_batch is None")
            if self.return_answer_sentence:
                print("return_answer_sentence:",batch_best_answers, batch_best_answers_score,
                      batch_best_answers_place, batch_best_answers_sentences)
                
                return (batch_best_answers, batch_best_answers_score, batch_best_answers_place,
                        batch_best_answers_sentences)
            
            print("*** Default Return: ",batch_best_answers, batch_best_answers_score, batch_best_answers_place)
            return batch_best_answers, batch_best_answers_score, batch_best_answers_place

        if self.return_answer_sentence:
            
            # 22/12/23 DH:
            """
            print("return_answer_sentence: len =",len(batch_best_answers[0]))
            for i in range(len(batch_best_answers[0])):
                print(i,") ",batch_best_answers[0][i],  " | ", batch_best_answers_score[0][i],  " | ",
                      batch_best_answers_place[0][i],  " | ", batch_best_answers_doc_ids[0][i],  " | ", 
                      batch_best_answers_sentences[0][i])
            """
            
            return (batch_best_answers, batch_best_answers_score, batch_best_answers_place,
                    batch_best_answers_doc_ids, batch_best_answers_sentences)
        
        print("Default return: ",batch_best_answers, batch_best_answers_score, batch_best_answers_place,
              batch_best_answers_doc_ids)
        return (batch_best_answers, batch_best_answers_score, batch_best_answers_place,
                batch_best_answers_doc_ids)
