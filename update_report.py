import json

# Read the notebook
with open('hmm_tagger.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

# New report text (more student-like, simpler language)
new_text = """Data Preparation Method

I created the dataset by combining web.conllu and wiki.conllu files. Both files are in CONLL-U format, which means each line has a word and its POS tag. I used the read_conll function to load the files and I removed lines that start with "#" because those are just metadata and not useful for training. I also removed empty lines. I stored each sentence as a list of (word, POS_tag) pairs to keep things organized. Then I made two different datasets: one with all the POS tags from the original files, and another simpler one with only ADJ, ADV, NOUN, VERB, and PUNCT tags. I used the prepare_data function to filter the data, which made it easy to get the same datasets for both models. After that, I split the data 80/20 for training and testing, and I used random_state=42 so I could get the same results every time I run it.

MLE Probability Calculation Method

For the Hidden Markov Model, I calculated transition and emission probabilities using Maximum Likelihood Estimation (MLE). To get transition probabilities, I counted how many times one tag came after another tag, and then I divided that by how many times the previous tag appeared in total. For emission probabilities, I divided how many times a word appeared with a certain tag by how many times that tag appeared overall. I had problems with zero probabilities, especially for rare words and tags, so I used Laplace smoothing with α=1 to fix this. I also added START and END tokens to handle the beginning and end of sentences properly, because these transitions are important for predicting sequences.

Challenges Faced

While I was working on this, I ran into several problems. One big issue was that when I multiplied lots of small probabilities in the Viterbi algorithm, I got numerical underflow errors. To fix this, I switched to using log-probabilities instead, which made the calculations work properly. Another problem was dealing with words that I never saw in the training data (OOV words). For these words, I gave them a small but non-zero emission probability. Turkish language also made things harder because it has an agglutinative structure. This means there are many different word forms, so the vocabulary gets really big and I see more unknown words. Some sentences in the dataset were also very long, which made the Viterbi matrix bigger and took more time to compute.

Performance Comments

I evaluated the model using accuracy and F1-score. The model I trained with all POS tags usually had lower accuracy because it had to choose between many different labels, but it gave more detailed tagging results. The model with fewer tags (only 5 tags) got higher accuracy and F1 scores because it had fewer classes to choose from. When I looked at the confusion matrix, I saw that most mistakes happened between tags that are similar to each other in meaning or how they are used.

Limitations and Possible Improvements

Right now my model only looks at bigram tag dependencies (one tag to the next). If I used trigram models, it could understand context better but it would also take more time to compute. My strategy for handling unknown words is pretty simple and I could make it better by using suffix patterns, character n-grams, or pretrained word embeddings. Since Turkish has a lot of word forms, adding CONLL-U features like Case or PersonNumber could make the accuracy much better. Other things I could try include testing different smoothing methods or using ensemble approaches where I combine multiple models."""

# Update cell 20 (the report cell)
cell = data['cells'][20]
# Split the new text into lines and add newline characters
lines = new_text.split('\n')
cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

# Write back to file
with open('hmm_tagger.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=1)

print("Report updated successfully!")
print(f"New text length: {len(new_text)} characters")

