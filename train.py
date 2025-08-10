import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# stopwords + stemming
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words_en = set(stopwords.words('english'))


def clean_text(text, remove_stopwords=True, stem_english=True):
    #supports unicode rather than [^a-zA-Z]
    text = re.sub(r'[^\p{L}\s]', ' ', str(text))
    text = text.lower()
    words = text.split() #tokenizing
    cleaned_words = []
    for word in words:
        if remove_stopwords and word in stop_words_en:
            continue
        if stem_english and re.match(r'^[a-z]+$', word):
            word = ps.stem(word)
        cleaned_words.append(word)
    return ' '.join(cleaned_words)

data = pd.read_csv('C:/Users/elinz/OneDrive/Python Projects/Language Detection/data.csv')

corpus = [clean_text(t) for t in data['Text']]

# converts to numeric features
#n-grams => lenght of sequence
cv = CountVectorizer(analyzer='char', ngram_range=(2,4), max_features=10000)
X = cv.fit_transform(corpus).toarray()

#target language labels
label = LabelEncoder()
y = label.fit_transform(data['Language'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = MultinomialNB()
classifier.fit(x_train, y_train)

train_pred = classifier.predict(x_train)
test_pred = classifier.predict(x_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

#plot training results
#rows = actual classes
#cols = predicted classes
cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label.classes_,
            yticklabels=label.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Set')
plt.show()

joblib.dump((classifier, cv, label, ps, stop_words_en), 'language_prediction.sav')
print("Model training complete and saved as 'language_prediction.sav'")
