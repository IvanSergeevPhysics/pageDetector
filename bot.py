import os
import asyncio
import logging
import extractPage
from aiogram import executor
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import Command
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from utils.set_bot_commands import set_default_commands
from aiogram.utils.exceptions import PhotoDimensions

class Extract(StatesGroup):
    extraction = State()

TOKEN = os.environ.get("BOT_TOKEN")

logging.basicConfig(level=logging.INFO, filename="bot_log.log", filemode="w")

bot = Bot(token = TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


async def on_startup(dispatcher):
    await set_default_commands(dispatcher)

async def send_message_loading(chat_id):
    message = await bot.send_message(chat_id, 'Loading results...')
    return message

@dp.message_handler(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hi! I am PageDetector. You can send me a photo with document and I will send you a scan of it." +
                         "You can also mark the photo with the format you want to get (jpg as default).")

@dp.message_handler(Command("help"))
async def cmd_start(message: types.Message):
    text = ("List of commands: ",
            "/start - Start using bot",
            "/help - Get list of commands",
            "/github - Get link to the github repo of the project")
    await message.answer("\n".join(text))

@dp.message_handler(Command("github"))
async def cmd_start(message: types.Message):
    text = "Link to the github: https://github.com/IvanSergeevPhysics/pageDetector"
    await message.answer(text)

@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):

    try:
        message_ = await send_message_loading(message.chat.id)

        await message.photo[-1].download('image.jpg')
        format = message.caption
        if format == None or format.replace(" ", "").lower() not in ['jpg', 'png', 'pdf']:
            format = 'jpg'
        extractPage.process(image_format=format)
        await bot.delete_message(message.chat.id, message_.message_id)
        await bot.send_document(message.chat.id, types.InputFile(f'page.{format}'))
        os.remove(f'page.{format}')
        os.remove('image.jpg')
    except PhotoDimensions:
        await message.answer("Your photo is so huge... Please send me it by file")
    

@dp.message_handler(content_types=types.ContentType.DOCUMENT)
async def handle_docs_doc(message: types.Document):

    message_ = await send_message_loading(message.chat.id)
    
    await message.document.download('image.jpg')
    format = message.caption
    if format == None or format.replace(" ", "").lower() not in ['jpg', 'png', 'pdf']:
        format = 'jpg'
    extractPage.process(image_format=format)
    await bot.delete_message(message.chat.id, message_.message_id)
    await bot.send_document(message.chat.id, types.InputFile(f'page.{format}'))
    os.remove(f'page.{format}')
    os.remove('image.jpg')


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    executor.start_polling(dp, on_startup=on_startup)