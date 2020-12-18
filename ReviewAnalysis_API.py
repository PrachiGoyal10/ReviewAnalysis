import nltk
import re
import pandas as pd
import os
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag_sents
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def show_wordcloud_fn (reviewdata,title=None) :
    # data = re.sub(r'\\W',"",str(data))
    wordcloudimg = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(str(reviewdata).replace("'", "").replace(",", "").replace('"', ''))

    fig = plt.figure(1, figsize=(20, 20))
    plt.axis('off')
    if title :
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    img = BytesIO()
    image = wordcloudimg.to_image()
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode('utf-8')

def stp(corpus):
    stop_words = nltk.corpus.stopwords.words('english')
    words = nltk.word_tokenize(corpus)
    word_s = [word for word in words if word not in stop_words]
    corpus= ' '.join(word_s)
    return corpus


def lemma (corpus) :
    lemmatizer = WordNetLemmatizer()
    lemsentence = []
    for sen in corpus :
        words = nltk.word_tokenize(sen)
        word_S = [lemmatizer.lemmatize(word) for word in words]
        w = ' '.join(word_S)
        lemsentence.append(w)
    return lemsentence

def tagme(corpus):
    texts = corpus.tolist()
    tagged_texts = pos_tag_sents(map(word_tokenize, texts))
    return tagged_texts

#####################UI feaure word based search on review ####################
def getreviewforword (df,word) :
    sent = []
    for reviews in df['Review_Text'] :
        for words in reviews.split(" ") :
            if word.lower() == words.lower() :
                sent.append(reviews)
    return sent


def getFeature(foldername):
    filenamelist = []
    # foldername = 'ratings2020'
    for subdir, dirs, files in os.walk(foldername) :
        for file in os.listdir(subdir) :
            filepath = subdir + os.sep + file
            re.sub(r"\\", "/", filepath)
            if ".csv" in filepath :
                filenamelist.append(filepath)

    # ----------------> Merging all the data in one csv
    df_merged = (
    pd.read_csv(filepath_or_buffer=file, sep=',', encoding='utf-16', error_bad_lines=False, engine='python') for file in
    filenamelist)
    df_merged = pd.concat(df_merged, ignore_index=True)
    df_merged.to_csv("merged.csv")
    df_merged.columns = [column.replace(" ", "_") for column in df_merged.columns]
    df = df_merged[["Star_Rating", "Reviewer_Language", "Review_Text", "App_Version_Code"]]
    pd.set_option('mode.chained_assignment', None)  # to remove SettingwithcopyWarning

    df['Positively_Rated'] = np.where(df['Star_Rating'] >= 3, 1, 0)
    # @@@@@@@@@@@@@@@@@@@ UI FEATURE 1: @@@@@@@@@@@@@@@@@@@@@@@@@@
    total_rating = len(df['Star_Rating'])
    pd.set_option('mode.chained_assignment', None)
    df.dropna(inplace=True, how='any')
    total_reviews = len(df(l1['Review_Text']))

    # In version 1.0 , we'll be checking only english revviews....
    df = df[df.Reviewer_Language == 'en']

    # Telling the positive and negative Cont and propotion for a particular version
    latest_version = max(df["App_Version_Code"])
    VrsnRating = df[df.App_Version_Code == latest_version].Positively_Rated.mean()

    VrsnRating = round(VrsnRating*100,2)


    ########## DATA CLEANING ##################333
    df['Review'] = df['Review_Text'].apply(lambda x : x.lower())
    df['Review'] = df['Review'].apply(lambda x : re.sub(r"\W", " ", x))  # non -word charactrer
    df['Review'] = df['Review'].apply(lambda x : re.sub(r"\d", " ", x))  # removing digits
    df['Review'] = df['Review'].apply(lambda x : re.sub("([^\x00-\x7F])+", " ", x))  # removing emojis
    df['Review'] = df['Review'].apply(lambda x : re.sub(' \w{1,4} ', ' ', x))  # removing 2  char wrds
    df['Review'] = df['Review'].apply(lambda x : re.sub(r"\s+", " ", x))
    df['Review'] = lemma(df['Review'])
    df['Review'] = df['Review'].apply(stp)
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(inplace=True)
    df.isnull()
    df['Review'] = tagme(df['Review'])

    sid = SentimentIntensityAnalyzer()
    df["sentiments"] = df["Review_Text"].apply(lambda x: sid.polarity_scores(x))  #'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound':..
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

    # add number of characters column
    df["nb_chars"] = df["Review_Text"].apply(lambda x : len(x))
    # add number of words column
    df["nb_words"] = df["Review_Text"].apply(lambda x : len(x.split(" ")))

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["Review"].apply(lambda x : str(x).split(" ")))]
    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=30, window=2, min_count=1, workers=4)
    # transform each document into a vector data
    doc2vec_df = df["Review"].apply(lambda x : model.infer_vector(str(x).split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    df = pd.concat([df, doc2vec_df], axis=1)

    corpus = []
    for sentences in df["Review"] :
        corpus.append([word for word, tag in sentences])

    df['cln_Reviews'] = [" ".join(review) for review in corpus]

    # add tf-idfs columns
    tfidf = TfidfVectorizer(min_df=5)  # ignore terms appearing less than 5 documents
    tfidf_result = tfidf.fit_transform(df["cln_Reviews"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = df.index
    reviews_df = pd.concat([df, tfidf_df], axis=1)

    wrdcldimg = show_wordcloud_fn(corpus)

    best_negsentences = reviews_df[reviews_df["nb_words"] >= 5].sort_values("neg", ascending=False)[["Review_Text"]].head()
    #best_negsentences = reviews_df.sort_values("neg", ascending=False)[["Review_Text"]].head()
    best_negsentences = best_negsentences.to_string(index=False)


    pos_best_sentences = reviews_df[reviews_df["nb_words"] >= 5].sort_values("pos", ascending=False)[["Review_Text"]].head()
    #pos_best_sentences = reviews_df.sort_values("pos", ascending=False)[["Review_Text"]].head()
    pos_best_sentences = pos_best_sentences.to_string(index=False)

    # apprtngimg = appvsrating(reviews_df)

    return (best_negsentences, pos_best_sentences,total_rating,total_reviews,VrsnRating,latest_version,wrdcldimg)








#
#
# best_negsentences, pos_best_sentences,total_rating,total_reviews,VrsnRating,latest_version,wordcloud = getFeature(r'ratings2020')
# print(best_negsentences)
# print("####################")
# print(pos_best_sentences)
# print(type(wordcloud))
#
# print(total_rating,total_reviews,VrsnRating,latest_version)
#
#
# for i in best_negsentences.split("\n"):
#     print(i)
#
# for i in pos_best_sentences.split("\n"):
#     print(i)
