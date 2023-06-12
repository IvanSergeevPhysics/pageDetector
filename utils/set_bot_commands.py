from aiogram import types


async def set_default_commands(dp):
    await dp.bot.set_my_commands(
        [
            types.BotCommand("start", "Get bot started"),
            types.BotCommand("help", "Get the list of commands"),
            types.BotCommand("github", "Get link to the github repo of the project")
        ]
    )
