# Language-Prediction-Model
A simple language detection tool that uses character-level n-grams combined with a Multinomial Naive Bayes classifier. It preprocesses text by cleaning, removing stopwords, and applying stemming before training a model to predict the language of given text input.


## Features

- Unicode-aware text cleaning  
- Stopword removal and Porter stemming (English)  
- Character n-gram vectorization (2 to 4 characters)  
- Multinomial Naive Bayes classification  
- Model saving and loading with `joblib`  
- Interactive command-line interface for language prediction  


## Example Output
```bash
Type a phrase and I'll guess its language! (type 'exit' to quit)
Input: Hej, jag heter Bengt jag jobbar på en knappfabrik
Predicted Language: Swedish
Input: I love Labubus with my whole heart
Predicted Language: English
Input: exit
