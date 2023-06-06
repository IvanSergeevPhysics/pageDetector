import os
import asyncio
import logging
import extractPage
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import Command
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from utils.set_bot_commands import set_default_commands

class Extract(StatesGroup):
    extraction = State()

TOKEN = '12345'

logging.basicConfig(level=logging.INFO, filename="bot_log.log", filemode="w")

bot = Bot(token = TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


async def on_startup(dispatcher):
    await set_default_commands(dispatcher)

@dp.message_handler(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hi! I am PageDetector. You can send me a photo with document and I will send you a scan of it:)")


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):

    await message.photo[-1].download('image.jpg')
    # cmd = f'python extractPage.py'
    # os.system(cmd)
    extractPage.process()
    await bot.send_photo(message.chat.id, types.InputFile('page.jpg'))
    os.remove('page.jpg')
    os.remove('image.jpg')


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())