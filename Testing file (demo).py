"""

import pandas as pd

# Read the Excel file
file_path6 = 'token_data_map.xlsx'  # Replace with your file path
df = pd.read_excel(file_path6, usecols="A:C")  # Read columns A and B

#df1=df.to_pandas()

# Display the data (optional)
print("Original Data:")
#print(df1[0])

print("First Row Data:")
print(df.iloc[0, :])  

#print("First Row Data:")
#print(df.iloc[1, :]) 

aa=df.iloc[1, :]

print(aa["ner"],aa["pos"])

print("Original Data2:")
print(df)
# Modify the data if needed (for example, append a new row)



#df.loc[len(df)] = ['token', 'ner','pos']
#df.loc[2] = ['token2', 'ner2','pos2']



# Write the modified data back to a new Excel file
#output_file_path = 'token_data_map.xlsx'  # Replace with desired output path

#df.to_excel(file_path6, index=False)

print(f"Data has been written to {file_path6}")


""" 



"""
from deep_translator import GoogleTranslator

# Set source language to Bengali ('bn') and target language to English ('en')
translator = GoogleTranslator(source='bn', target='en')
translated = translator.translate("হ্যালো")
print(translated)

"""


"""
import spacy

# Load pre-trained language model
nlp = spacy.load("en_core_web_md")

def are_words_similar(word1, word2, threshold=0.7):
    # Process words using spaCy's NLP model
    token1 = nlp(word1)
    token2 = nlp(word2)
    
    # Calculate similarity (returns a float between 0 and 1)
    similarity = token1.similarity(token2)
    
    # Check if similarity exceeds the threshold
    return similarity >= threshold, similarity

# Example usage
word1 = "furious"
word2 = "angry"

are_similar, similarity_score = are_words_similar(word1, word2)
print(f"Are the words similar? {are_similar} (Similarity score: {similarity_score})")


"""

'''
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_word_embedding(word):
    # Tokenize input word and get input IDs
    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
    
    # Get hidden states from the BERT model
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state
    
    # The embedding of the [CLS] token can be used as the word's embedding
    return hidden_states.mean(dim=1)

def are_words_similar_bert(word1, word2, threshold=0.7):
    # Get word embeddings
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    
    # Calculate cosine similarity
    similarity = torch.cosine_similarity(embedding1, embedding2).item()
    
    # Return whether the similarity exceeds the threshold
    return similarity >= threshold, similarity

# Example usage
word1 = "happy"
word2 = "joyful"
are_similar, similarity_score = are_words_similar_bert(word1, word2)
print(f"Are the words similar? {are_similar} (Similarity score: {similarity_score})")
'''



import json

with open("translation_data//banglat5_translations.json", "r", encoding="utf-8") as f1:
    bangla1_data = json.load(f1)
    
    
#print(bangla1_data)

with open("translation_data//nllb_translations.json", "r", encoding="utf-8") as f2:
    bangla2_data = json.load(f2)
    
    
    
    
def dataset_gen():
    from datasets import load_from_disk
    dataset = load_from_disk("data")
    print(dataset)
    
    return dataset
    
dataset=dataset_gen()
#dataset.head(10)

for i in range(30):
    print(dataset[i])


'''
# Install the bnltk library for Bangla language processing if you haven't already.
# !pip install bnltk

from bnltk.stemmer import BanglaStemmer

# Initialize the Bangla stemmer
stemmer = BanglaStemmer()

# Input: Bangla words (change these to the words you want to stem)
words = ['কুকুরগুলোকে', 'বইটির', 'গাছটির']

# Stemming each word in the list
for word in words:
    root_word = stemmer.stem(word)
    print(f'Original word: {word} --> Root word: {root_word}')

'''

"""
import bangla_stemmer

# Input: Bangla words
words = ['কুকুরগুলোকে', 'বইটির', 'গাছটির']

# Stemming each word using the module's functionality
for word in words:
    root_word = bangla_stemmer.stem(word)  # Adjust this based on what the module exposes
    print(f'Original word: {word} --> Root word: {root_word}')

"""

'''
from banglanlp import Lemmatizer

# Initialize the Bangla lemmatizer
lemmatizer = Lemmatizer()

# Input: Bangla words
words = ['কুকুরগুলোকে', 'বইটির', 'গাছটির']

# Lemmatizing each word
for word in words:
    root_word = lemmatizer.lemmatize(word)
    print(f'Original word: {word} --> Root word: {root_word}')

'''

'''
from bnltk.stemmer import BanglaStemmer

# Initialize the Bangla stemmer
stemmer = BanglaStemmer()

# Input: Bangla words
words = ['কুকুরগুলোকে', 'বইটির', 'গাছটির']

# Stemming each word
for word in words:
    root_word = stemmer.stem(word)
    print(f'Original word: {word} --> Root word: {root_word}')

'''