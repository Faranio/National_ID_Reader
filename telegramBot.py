from telebot.types import Message
from main import run
from PIL import Image

import os
import telebot
import time
import torchvision

TOKEN = "944752417:AAGiWnpHHpDwV6Za1t3xXgmy7PfSoaxFy2U"
bot = telebot.TeleBot(TOKEN)
front = False
resnet = torchvision.models.resnet152(pretrained=True, progress=True).cuda()


@bot.message_handler(commands=['start'])
def start(message: Message):
    bot.reply_to(message,
                 'Hello and welcome! Instructions on how to use this bot:\n 1. Send the /front or /back command\n 2. Send the image of an ID card with the corresponding side\n 3. Wait for the answer')


@bot.message_handler(commands=['help'])
def help(message: Message):
    bot.reply_to(message,
                 'This bot responds only to these commands:\n /front - setting the front side argument to True\n /back - setting the front side argument to False\n /check - checking which side is currently turned on.')


@bot.message_handler(content_types=['text'])
def conversation(message: Message):
    global front
    if message.text == "/front":
        front = True
        bot.reply_to(message, "Front: {}. Now send me the photo of an ID card in a proper position (not rotated)...".format(front))
    elif message.text == "/back":
        front = False
        bot.reply_to(message, "Front: {}. Now send me the photo of an ID card in a proper position (not rotated)...".format(front))
    elif message.text == "/check":
        if front:
            bot.reply_to(message, "Front side is turned on.")
        else:
            bot.reply_to(message, "Back side is turned on.")
    else:
        bot.reply_to(message, "Error! There are only 3 commands: /front, /back and /check!")


@bot.message_handler(content_types=['photo'])
def photo(message: Message):
    raw = message.photo[2].file_id
    path = raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(path, 'wb') as new_file:
        new_file.write(downloaded_file)
    answer = run(path, resnet=resnet, front=front)
    bot.reply_to(message, answer)
    os.remove(path)


while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print(e)
        time.sleep(15)
