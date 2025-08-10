import regex as re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

model, cv, label, ps, stop_words_en = joblib.load('language_prediction.sav')

def clean_text(text, remove_stopwords=True, stem_english=True):
    text = re.sub(r'[^\p{L}\s]', ' ', str(text))
    text = text.lower()
    words = text.split()
    cleaned_words = []
    for word in words:
        if remove_stopwords and word in stop_words_en:
            continue
        if stem_english and re.match(r'^[a-z]+$', word):
            word = ps.stem(word)
        cleaned_words.append(word)
    return ' '.join(cleaned_words)

def test_model(test_sentence):
    cleaned = clean_text(test_sentence)
    vec = cv.transform([cleaned]).toarray()
    pred = model.predict(vec)[0]
    language = label.inverse_transform([pred])[0]
    print(f"Predicted Language: {language}")

def main():
    print("Type a phrase and I'll guess its language! (type 'exit' to quit)")
    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            break
        test_model(user_input)

if __name__ == "__main__":
    main()
