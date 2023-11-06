### Author: Riya Nakarmi ###
### College Project ###

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def process(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res


from langdetect import detect
from deep_translator import GoogleTranslator


# simple function to detect and translate text
def detect_and_translate(text, target_lang):
    result_lang = detect(text)

    if result_lang == target_lang:
        return text

    else:
        translator = GoogleTranslator(source=result_lang, target=target_lang).translate(text)
        return translator




from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    result_lang = detect(userText)
    if result_lang == 'en':
        return (str(process(userText)))
    else:
        translator = GoogleTranslator(source=result_lang, target='en').translate(userText)
        response = str(process(translator))
        responseTranslator = GoogleTranslator(source='en', target=result_lang).translate(response)
        return (responseTranslator)

@app.route('/other_folder')
def other_folder():
    import openai
    import gradio

    openai.api_key = "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"

    messages = [{"role": "system",
                 "content": "You are a financial experts that specializes in real estate investment and negotiation"}]

    def CustomChatGPT(user_input):
        messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        ChatGPT_reply = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": ChatGPT_reply})
        return ChatGPT_reply

    demo = gradio.Interface(fn=CustomChatGPT, inputs="text", outputs="text", title="scientific questions")

    demo.launch(share=True)
    return render_template('other_folder.html')




 

if __name__=="__main__":
    app.run(debug=True)
