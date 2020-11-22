import telepot
import time
from PIL import Image


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

TOKEN="Your Telegram token"

bot = telepot.Bot(TOKEN)
bot.getMe()

from pprint import pprint
resp = bot.getUpdates(offset=-1)
pprint(resp)

def make_answer(message):
    sentence = tokenize(message)#Tokenize---------

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    answer = ''

    if prob.item()>0.65:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                answer = (str(intent['responses'][0]))
    else:
        answer = 'Non ho capito, cerca di essere più specifico...'

    return answer



# funzione che viene eseguita ad ogni messaggio ricevuto
def handle(message):
    content_type, chat_type, chat_id = telepot.glance(message)
    answer = make_answer(message['text'])
    if message['text'] == '/start':
        bot.sendMessage(chat_id, "Ciao, sono qui per aiutarti, fammi una domanda e cercherò di esserti d'aiuto. \nRicorda di specificare il ruolo che occupi (professore, alunno, vicario), così potrò essere più preciso")
        #bot.sendPhoto(chat_id, (open('/Users/andrea/Desktop/chestnut/Anagramma/anagramma.png', "rb")))
    else:
        bot.sendMessage(chat_id, answer)
 
bot.message_loop(handle)
print ('In attesa di nuovi messaggi...')
while 1:
    time.sleep(10)
