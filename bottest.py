#environment: ml_bot
#username: ppprojectml_bot
#token: 6990039802:AAF-Tu_iyfGCKczk4g8ZDRotgF5RbnsUHUs
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os

import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from predict import result

TOKEN: Final = '6990039802:AAF-Tu_iyfGCKczk4g8ZDRotgF5RbnsUHUs'
BOT_USERNAME: Final = '@ppprojectml_bot'

with open('intents.json','r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()

#Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Let's begin.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("I am an image captioner, please provide me an image so I can generate a caption in arabizi language.")
async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("This is a custom command.")

async def handle_photos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if filters.PHOTO.check_update(update):
        file_id = update.message.photo[-1].file_id
        unique_file_id = update.message.photo[-1].file_unique_id
        photo_name = f"{unique_file_id}.jpg"
    elif filters.Document.IMAGE and update.message.document.file_id:
        file_id = update.message.document.file_id
        _, f_ext = os.path.splitext(update.message.document.file_name)
        unique_file_id = update.message.document.file_unique_id
        photo_name = f"{unique_file_id}.{f_ext}"
    photo_file = await context.bot.get_file(file_id)
    downloaded_path = f'./temp/{photo_name}'
    await photo_file.download_to_drive(custom_path=downloaded_path)
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Yalla 3atineh di2a...')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=result(downloaded_path))

#Responses
async def handle_message(update, context):
    user_input = update.message.text
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                await update.message.reply_text(response)
    else:
        await update.message.reply_text("Sorry, ma fhemet 3alek...")

#used for logging
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    #Commands
    app.add_handler(CommandHandler('start',start_command))
    app.add_handler(CommandHandler('help',help_command))
    app.add_handler(CommandHandler('custom',custom_command))

    #Messages
    app.add_handler(MessageHandler(filters.TEXT,handle_message))
    app.add_handler(MessageHandler(filters.Document.IMAGE | filters.PHOTO, handle_photos))

    #Errors
    app.add_error_handler(error)

    #Polls the bot
    print('Polling...')
    app.run_polling(poll_interval=3) #check every 3 seconds for messages