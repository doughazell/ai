from transformers import pipeline

# 13/1/24 DH:
print("--------------------------------")
print("Using Transformers 'pipeline'")
print("--------------------------------")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

with open("xl-bully-ban-short.txt") as source :
  ARTICLE = source.readlines()

summaryList = summarizer(ARTICLE, max_length=130, min_length=10, do_sample=False)

with open("pipeline-out.txt", "w") as fout:
  print()
  for item in summaryList:
    summary_text = item['summary_text']

    if "CNN.com" not in summary_text:
      print(summary_text)
      fout.write(summary_text + "\n")
    else:
      print("  *** REMOVING: 'CNN.com will feature...' advert ***")

  #fout.flush()


