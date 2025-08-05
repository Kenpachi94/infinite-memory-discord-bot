import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import logging

import socket
print("DNS Check:", socket.gethostbyname('discord.com'))

from datetime import datetime
import discord
from discord import app_commands
from discord.ext import commands
import openai
from openai import OpenAI
import psycopg

import aiohttp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord-bot")

# Load environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

openai.api_key = OPENAI_API_KEY

# Discord Bot Setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

from psycopg_pool import AsyncConnectionPool

db_pool = None  # will initialize in on_ready

# Ensure table exists for storing messages
CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS discord_messages (
    server_id TEXT,
    channel_id TEXT,
    username TEXT,
    content TEXT,
    is_bot BOOLEAN,
    timestamp TIMESTAMPTZ DEFAULT now()
);
"""

@bot.event
async def on_ready():
    global db_pool
    logger.info(f"Bot logged in as {bot.user.name}")
    db_pool = AsyncConnectionPool(DATABASE_URL, max_size=5, timeout=10)
    await db_pool.open()  # ✅ Open the pool here safely
    await bot.tree.sync()

    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(CREATE_MESSAGES_TABLE)
            await conn.commit()


# Store every message from server
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if isinstance(message.channel, discord.DMChannel):
        return  # skip DMs entirely

    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO discord_messages (server_id, channel_id, username, content, is_bot)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    str(message.guild.id),
                    str(message.channel.id),
                    message.author.name,
                    message.content,
                    False,
                )
            )
            await conn.commit()

    await bot.process_commands(message)


@bot.tree.command(name="chat", description="Ask a question with full server context")
@app_commands.describe(question="Your question")
async def chat(interaction: discord.Interaction, question: str):
    if isinstance(interaction.channel, discord.DMChannel):
        await interaction.response.send_message("This command is only available in servers.", ephemeral=True)
        return

    await interaction.response.defer()

    server_id = str(interaction.guild_id)
    channel_id = str(interaction.channel_id)

    # ✅ Store user's slash command input in DB
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO discord_messages (server_id, channel_id, username, content, is_bot)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    server_id,
                    channel_id,
                    interaction.user.name,
                    question,
                    False,
                )
            )
            await conn.commit()

    # Fetch full chat history from server and channel
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT username, content FROM discord_messages
                WHERE server_id = %s AND channel_id = %s
                ORDER BY timestamp ASC
                """,
                (server_id, channel_id)
            )
            rows = await cur.fetchall()

    # Build messages context
    context_str = "\n\n".join([
        f"{row[0]}: {row[1]}" for row in rows
    ])

    system_prompt = '''Spiritual chat bot with deep knowledge in spirituality, occult, esoteric, history, energy'''

    prompt = f"""Conversation history:
{context_str}

Current question: {question}

Respond in alignment with the style and context."""

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            top_p=0.9,
            max_tokens=2000,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        answer = response.choices[0].message.content

        # Store bot's message
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO discord_messages (server_id, channel_id, username, content, is_bot)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        server_id,
                        channel_id,
                        bot.user.name,
                        answer,
                        True,
                    )
                )
                await conn.commit()

        await interaction.followup.send(answer)

    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        await interaction.followup.send(f"Sorry, I encountered an error: {str(e)}")


async def ensure_db_ready():
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
    except psycopg.OperationalError:
        logger.warning("DB connection stale, reopening pool.")
        await db_pool.close()
        await db_pool.open()

if __name__ == "__main__":
    if not DISCORD_TOKEN or not OPENAI_API_KEY or not DATABASE_URL:
        raise RuntimeError("DISCORD_TOKEN, OPENAI_API_KEY and PIXELTABLE_DB_URL must be set.")
    bot.run(DISCORD_TOKEN)
