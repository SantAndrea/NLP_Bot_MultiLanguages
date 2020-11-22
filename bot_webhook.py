import telebot
from flask import Flask, request
import os

import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#-------------------------- 

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

#-------------------------


TOKEN = ''
bot = telebot.TeleBot(token=TOKEN)
app = Flask(__name__)


@bot.message_handler(commands=['start']) # welcome message handler
def send_welcome(message):
    bot.send_message(chat_id=message.chat.id, text='Ciao')


@bot.message_handler(func=lambda message: True)
def echo_message(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item()>0.65:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                s = (f"{bot_name}: {intent['responses']}")
                bot.send_message(chat_id=message.chat.id, text=s)
    else:
        bot.send_message(chat_id=message.chat.id, text='Non ho capito, cerca di essere più specifico...')
    



@app.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@app.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='yourURL' + TOKEN)
    return "!", 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')