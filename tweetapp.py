from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    text1 = request.form['text1'].lower()

    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])
    
    result = request.form
    sa = SentimentIntensityAnalyzer()
    s_dic=sa.polarity_scores(result)
    sentiment=None
    if s_dic['compound']>0:
          sentiment='positive'
    elif s_dic['compound']<=0:
          sentiment='negative'
    else:
          sentiment=='neutral'

    return render_template('form.html', final=sentiment, text1=text1)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
