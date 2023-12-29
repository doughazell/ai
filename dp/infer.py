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
import json
import sys
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

from deeppavlov.core.commands.utils import import_packages, parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.download import deep_download
from deeppavlov.utils.pip_wrapper import install_from_config

log = getLogger(__name__)


def build_model(config: Union[str, Path, dict], mode: str = 'infer',
                load_trained: bool = False, install: bool = False, download: bool = False) -> Chainer:
    """Build and return the model described in corresponding configuration file."""
    config = parse_config(config)

    if install:
        install_from_config(config)
    if download:
        deep_download(config)

    import_packages(config.get('metadata', {}).get('imports', []))

    model_config = config['chainer']

    print("**************************************")
    print("*** build_model() - model_config['in']:", model_config['in'])
    print("**************************************")
    
    model = Chainer(model_config['in'], model_config['out'], model_config.get('in_y'))

    for component_config in model_config['pipe']:
        if load_trained and ('fit_on' in component_config or 'in_y' in component_config):
            try:
                component_config['load_path'] = component_config['save_path']
                print("load_path: ",component_config['load_path'])
            except KeyError:
                log.warning('No "save_path" parameter for the {} component, so "load_path" will not be renewed'
                            .format(component_config.get('class_name', component_config.get('ref', 'UNKNOWN'))))

        component = from_params(component_config, mode=mode)

        if 'id' in component_config:
            model._components_dict[component_config['id']] = component

        if 'in' in component_config:
            c_in = component_config['in']
            c_out = component_config['out']
            in_y = component_config.get('in_y', None)
            main = component_config.get('main', False)
            model.append(component, c_in, c_out, in_y, main)

    # 19/12/23 DH:
    print()
    print("=== build_model() - COMPLETED MODEL ===")
    print("model: ")
    for elem in model:
        print("  ",elem)
    print("-----------------------")
    
    return model


def interact_model(config: Union[str, Path, dict]) -> None:
    """Start interaction with the model described in corresponding configuration file."""
    model = build_model(config)

    # 29/12/23 DH: This needs to be defined outside 'while True' since it gets assigned on the return 
    #              from the previous question
    predDict = {}

    while True:
        args = []
        for in_x in model.in_x:
            args.append((input('{}::'.format(in_x)),))
            # check for exit command (in last arg of batch)
            if args[-1][0] in {'exit', 'stop', 'quit', 'q'}:
                return
            
        if checkForSaving(args, predDict) is True:
            # 28/12/23 DH: This 'continue' needs to be OUTSIDE THE FOR-LOOP of arg input above
            continue

        # 29/12/23 DH: Initiate the 'pipe' by calling 'Chainer.__call__()'
        print("interact_model() args: ",args)
        pred = model(*args)

        # 29/12/23 DH: Check + reset 'stop flag' (if it was set when Cache DB ids found)
        if Chainer.stopFlag is True:
            print()
            print("interact_model() - Chainer 'stopFlag' was set so resetting...")
            print()
            Chainer.stopFlag = False

            printCacheOutput()
        else:
            predDict = printPipeOutput(model, pred)

# 29/12/23 DH:
def checkForSaving(args, predDict) -> bool:
    # 28/12/23 DH: Args is array of tuples
    #              Args are designed to have batch questions so '-1' means last entry (with slice notation)
    import re
    
    #LITERAL SEARCH (not "wildcard"): if args[0][0] in {'save', 's', 'save *', 's *'}:

    # https://docs.python.org/3/howto/regex.html#performing-matches, "The r prefix, making the literal 
    # a raw string literal, is needed in this example because [of] escape sequences in a normal “cooked” 
    # string literal that are not recognized by Python"
    if re.match(r"s\b", args[0][0]) or re.match(r"save\b", args[0][0]):
        cmdArgs = (args[0][0]).split()

        if len(cmdArgs) > 1:
            try:
                i = int(cmdArgs[1])
                if i <= len(predDict['idList']):

                    record = {
                        'id': predDict['idList'][i-1],
                        'title': predDict['answerList'][i-1],
                        'text': predDict['sentenceList'][i-1]
                    }
                    print()
                    print("SAVING: id: ", record['id'], ", title: ", record['title'], ", text: ", record['text'])
                    print()

                    from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
                    SQLiteDataIterator.save_record(record)
            except ValueError:
                pass
            except UnboundLocalError:
                pass

            return True # ie continue with 'while True'

    return False

# 29/12/23 DH:
def printCacheOutput():
    print("Getting 'ngram' in 'printCacheOutput()' via 'StreamSpacyTokenizer._getLongestNGram()'")
    from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer

    ngram = StreamSpacyTokenizer._getLongestNGram()

    from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
    result = SQLiteDataIterator.getDetailsFromTitle(ngram)

    print()
    print(result)
    print()

    for (id, title, text) in result:
        print("{}) {}".format(id, text))

# 29/12/23 DH:
def printPipeOutput(model: Chainer, pred: str) -> dict:
    if len(model.out_params) > 1:
        # 27/12/23 DH: https://docs.python.org/3/library/functions.html#zip, "returns an iterator of tuples"
        # https://docs.python.org/3/glossary.html#term-iterator, "Attempting this with an iterator will just 
        # return the same exhausted iterator object used in the previous iteration pass, making it appear like 
        # an empty container."
        
        pred = zip(*pred)

    #print('>>', *pred)
    
    # 27/12/23 DH: "out": ["answer", "answer_score", "answer_place", "answer_id", "answer_sentence"]
    
    predList = list(pred)
    predDict = {}

    if predList:
        predDict['answerList'] = predList[0][0]
        predDict['scoreList'] = predList[0][1]
        predDict['placeList'] = predList[0][2]
        predDict['idList'] = predList[0][3]
        predDict['sentenceList'] = predList[0][4]

        print()
        print("-------------- interact_model() ---------------")
        
        for i in range(len(predList[0])):
            print(i+1,") ","id: ", predDict['idList'][i],", title: ", predDict['answerList'][i],
                  ", text: ", predDict['sentenceList'][i])

        print("-----------------------------------------------")
        print()
    
    return predDict


def predict_on_stream(config: Union[str, Path, dict],
                      batch_size: Optional[int] = None,
                      file_path: Optional[str] = None) -> None:
    """Make a prediction with the component described in corresponding configuration file."""

    batch_size = batch_size or 1
    if file_path is None or file_path == '-':
        if sys.stdin.isatty():
            raise RuntimeError('To process data from terminal please use interact mode')
        f = sys.stdin
    else:
        f = open(file_path, encoding='utf8')

    model: Chainer = build_model(config)

    args_count = len(model.in_x)
    while True:
        batch = list((l.strip() for l in islice(f, batch_size * args_count)))

        if not batch:
            break

        args = []
        for i in range(args_count):
            args.append(batch[i::args_count])

        res = model(*args)
        if len(model.out_params) == 1:
            res = [res]
        for res in zip(*res):
            res = json.dumps(jsonify_data(res), ensure_ascii=False)
            print(res, flush=True)

    if f is not sys.stdin:
        f.close()
