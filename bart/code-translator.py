# 14/1/24 DH:
from transformers.utils import is_torch_available

if is_torch_available():
  from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, AutoModelForSeq2SeqLM

from transformers import AutoTokenizer, BartForConditionalGeneration
import torch

# 14/1/24 DH:
print("--------------------------------")
print("Using Transformers 'pipeline'")
print("--------------------------------")

# 14/1/24 DH: 'transformers/pipelines/base.py' :
"""
class Pipeline(_ScikitCompat):

  Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following 
  operations: 
    Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

  @abstractmethod
  def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
    Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for
    `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
    
"""
# 14/1/24 DH: 'transformers/pipelines/text2text_generation.py' :
"""
class Text2TextGenerationPipeline(Pipeline):

  # At end of 'Tokenization' step
  def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
    inputs = self._parse_and_tokenize(inputs, truncation=truncation, **kwargs)
    return inputs

  def _forward(self, model_inputs, **generate_kwargs):
    ...
    output_ids = self.model.generate(**model_inputs, **generate_kwargs)

"""
# 14/1/24 DH: 'inputs' printed from 'pipeline.py' are 'BART-double-coded' values

# --------------------------------------------------------------------------------------------------------------
# 15/1/24 DH:
def paragraphSummary(filename):
  intStrList = []
  with open(filename) as source :
    for line in source.readlines():
      # Remove newline character from each int printout line
      lineStrip = line.rstrip()

      for item in lineStrip.split(","):
        # Remove whitespace
        itemStrip = item.strip()
        # Guard against empty string after last ',' on line
        if itemStrip:
          intStrList.append(itemStrip)

  # 'input_ids' needs to be 2-D array (prob for obfuscation reasons...)
  int2DList = [[int(item) for item in intStrList]]
  input_ids = torch.tensor(int2DList)

  print("\n",tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
  print()

  summary_ids = model.generate(input_ids, num_beams=2, min_length=0, max_length=130)
  print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# 16/1/24 DH:
import inspect
print("model.generate: ", model.generate.__qualname__)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

filename = "bart-double-codes1.txt"
paragraphSummary(filename)

filename = "bart-double-codes2.txt"
paragraphSummary(filename)
# --------------------------------------------------------------------------------------------------------------

#MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
#autoSeq2Seq = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
# 14/1/24 DH: "models/bart/modeling_bart.py:class BartForConditionalGeneration(BartPretrainedModel):"

#print("AutoModelForSeq2SeqLM: ",AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn") )
# OUTPUT: 
"""
BartForConditionalGeneration(
  (model): BartModel(
    (shared): Embedding(50264, 1024, padding_idx=1)
    (encoder): BartEncoder(
    )
    (decoder): BartDecoder(
    )
  )
  (lm_head): Linear(in_features=1024, out_features=50264, bias=False)
)
"""



