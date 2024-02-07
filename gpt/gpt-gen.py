# 1/2/24 DH:
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model = GPT2Model.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#text = "Replace me by any text you'd like."
#text = "Why has Russia invaded Ukraine?"
text = "What is tennis?"

#encoded_input = tokenizer(text, return_tensors='pt')
#output = model(**encoded_input)

encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

output_sequences = model.generate(
  input_ids=encoded_input,
  do_sample=True,
)
output_txt = tokenizer.decode(output_sequences[0], clean_up_tokenization_spaces=True)

print()
print("INPUT: ")
print(encoded_input)
print()
print("OUTPUT: ")
print(output_sequences)
print()
print("INPUT TXT: ", text)
print("OUTPUT TXT", output_txt)

