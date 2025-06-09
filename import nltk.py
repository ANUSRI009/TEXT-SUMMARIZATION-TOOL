import nltk
from transformers import T5ForConditionalGeneration,T5Tokenizer
model=T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer=T5Tokenizer.from_pretrained('t5-small')
def summarize_text(text):
  input_ids=tokenizer.encode("summarize:"+text,return_tensors="pt")
  output=model.generate(input_ids,max_length=100,min_length=30)
  summary=tokenizer.decode(output[0],skip_special_tokens=True)
  return summary

text="your long piece of text here...."
summary=summarize_text(text)
print(summary)