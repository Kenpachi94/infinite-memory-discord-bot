import os
import logging
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands
import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

from message_formatter import MessageFormatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixeltable-bot")

class PixelTableBot:
    def __init__(self):
        self.logger = logging.getLogger("pixeltable-bot")
        self.server_tables = {}
        self.user_tables = {}  # For DM conversations
        self.formatter = MessageFormatter()

        # Initialize bot with all required intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True

        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.setup_commands()
        self.logger.info("Bot initialization completed")

    @staticmethod
    @pxt.expr_udf
    def get_embeddings(text: str) -> np.ndarray:
        """Generate embeddings using sentence transformer"""
        return sentence_transformer(text, model_id='intfloat/e5-large-v2')

    def store_message(self, server_id: str, channel_id: str, username: str, content: str):
        """Store a server message"""
        try:
            if server_id not in self.server_tables:
                self.initialize_server_tables(server_id)

            messages_table = self.server_tables[server_id]['messages']
            messages_table.insert([{
                'server_id': server_id,
                'channel_id': channel_id,
                'username': username,
                'content': content,
                'timestamp': datetime.now()
            }])

            self.logger.info(f"Successfully stored message for server {server_id}")

        except Exception as e:
            self.logger.error(f"Failed to store message: {str(e)}")

    def store_dm_message(self, user_id: str, content: str, is_bot: bool):
        """Store a DM message"""
        try:
            if user_id not in self.user_tables:
                self.initialize_user_tables(user_id)

            messages_table = self.user_tables[user_id]['messages']
            messages_table.insert([{
                'user_id': user_id,
                'content': content,
                'is_bot': is_bot,
                'timestamp': datetime.now()
            }])

        except Exception as e:
            self.logger.error(f"Failed to store DM: {str(e)}")
            raise

    def initialize_server_tables(self, server_id: str):
        """Initialize server-specific tables"""
        try:
            server_dir = f'discord_bot_{server_id}'
            tables = {}

            # Ensure directory exists
            try:
                pxt.create_dir(server_dir)
            except Exception as e:
                if "already exists" not in str(e):
                    raise

            # Safely get or create 'messages'
            try:
                tables['messages'] = pxt.get_table(f'{server_dir}.messages')
            except Exception:
                tables['messages'] = pxt.create_table(
                    f'{server_dir}.messages',
                    {
                        'server_id': pxt.String,
                        'channel_id': pxt.String,
                        'username': pxt.String,
                        'content': pxt.String,
                        'timestamp': pxt.Timestamp
                    }
                )

            # Safely get or create 'sentences'
            try:
                tables['messages_view'] = pxt.get_table(f'{server_dir}.sentences')
            except Exception:
                tables['messages_view'] = pxt.create_view(
                    f'{server_dir}.sentences',
                    tables['messages'],
                    iterator=StringSplitter.create(
                        text=tables['messages'].content,
                        separators='sentence',
                    )
                )
                tables['messages_view'].add_embedding_index('text', string_embed=self.get_embeddings)

            # Safely get or create 'chat'
            try:
                tables['chat'] = pxt.get_table(f'{server_dir}.chat')
            except Exception:
                tables['chat'] = pxt.create_table(
                    f'{server_dir}.chat',
                    {
                        'server_id': pxt.String,
                        'channel_id': pxt.String,
                        'question': pxt.String,
                        'timestamp': pxt.Timestamp
                    }
                )

            self.server_tables[server_id] = tables
            self.setup_chat_columns(server_id)

        except Exception as e:
            self.logger.error(f"Failed to initialize server tables: {str(e)}")
            raise

    def initialize_user_tables(self, user_id: str):
        """Initialize user-specific tables for DMs"""
        try:
            if user_id in self.user_tables:
                return

            user_dir = f'discord_dm_{user_id}'
            tables = {}

            try:
                # Try to get existing tables first
                tables['messages'] = pxt.get_table(f'{user_dir}.messages')
                tables['messages_view'] = pxt.get_table(f'{user_dir}.sentences')
                tables['chat'] = pxt.get_table(f'{user_dir}.chat')
            except Exception:
                try:
                    pxt.create_dir(user_dir)
                except Exception as e:
                    if "already exists" not in str(e):
                        raise

                # Create tables for DMs with corrected schema
                tables['messages'] = pxt.create_table(
                    f'{user_dir}.messages',
                    {
                        'user_id': pxt.String,
                        'content': pxt.String,
                        'is_bot': pxt.Bool,
                        'timestamp': pxt.Timestamp
                    }
                )

                tables['messages_view'] = pxt.create_view(
                    f'{user_dir}.sentences',
                    tables['messages'],
                    iterator=StringSplitter.create(
                        text=tables['messages'].content,
                        separators='sentence',
                    )
                )

                tables['messages_view'].add_embedding_index('text', string_embed=self.get_embeddings)

                tables['chat'] = pxt.create_table(
                    f'{user_dir}.chat',
                    {
                        'user_id': pxt.String,
                        'question': pxt.String,
                        'timestamp': pxt.Timestamp
                    }
                )

            self.user_tables[user_id] = tables
            self.setup_chat_columns_dm(user_id)

        except Exception as e:
            self.logger.error(f"Failed to initialize user tables: {str(e)}")
            raise

    def setup_chat_columns(self, server_id: str):
        """Set up computed columns for server chat"""
        try:
            tables = self.server_tables[server_id]
            messages_view = tables['messages_view']
            chat_table = tables['chat']

            try:
                @messages_view.query
                def get_context(question_text: str):
                    sim = messages_view.text.similarity(question_text)
                    return (
                        messages_view
                        .where(sim > 0.3)
                        .order_by(sim, asc=False)
                        .select(
                            text=messages_view.text,
                            username=messages_view.username,
                            sim=sim
                        )
                        .limit(20)
                    )
                chat_table.add_computed_column(context=get_context(chat_table.question))
            except Exception as e:
                if "already exists" not in str(e):
                    self.logger.error(f"Error adding context column: {str(e)}")

            try:
                @pxt.udf
                def create_prompt(context: list[dict], question: str) -> str:
                    sorted_context = sorted(context, key=lambda x: x['sim'], reverse=True)
                    context_parts = []
                    for msg in sorted_context:
                        if msg['sim'] > 0.3:
                            relevance = round(float(msg['sim'] * 100), 1)
                            context_parts.append(
                                f"[Relevance: {relevance}%]\n"
                                f"{msg['username']}: {msg['text']}"
                            )

                    context_str = "\n\n".join(context_parts)

                    return f'''Previous conversation context from the server:
                    {context_str}

                    Current question: {question}

                    Important:
                    - Use context naturally without explicitly stating memory or recall
                    - Focus solely on answering the specific question asked
                    - Keep the response concise and to the point'''

                chat_table.add_computed_column(prompt=create_prompt(
                    chat_table.context,
                    chat_table.question
                ))
            except Exception as e:
                if "already exists" not in str(e):
                    self.logger.error(f"Error adding prompt column: {str(e)}")

            try:
                SYSTEM_PROMPT = '''You are a spiritually attuned assistant grounded in metaphysical knowledge, esoteric wisdom, philosophy and practical guidance for the soul's journey.

                🔮 Your Purpose:
                - Offer insights aligned with ancient spiritual traditions (e.g., Hermeticism, alchemy, Vedanta, Taoism)
                - Help users navigate inner transformation, energy work, and mystical practices
                - Respond calmly, poetically when appropriate, but always remain grounded and practical

                🌌 Principles:
                1. Contextual Awareness:
                - Remember prior spiritual concerns or experiences
                - Avoid repetition and build gently on past revelations

                2. Energetic Sensitivity:
                - Mirror the user's energy respectfully
                - Encourage reflection, not instruction
                - Offer guidance, not judgment

                3. Style & Tone:
                - Use a soft, poetic, yet clear tone
                - Reference archetypes, myth, symbols, or cosmic laws when relevant
                - Never assume too much—invite the seeker to explore further

                📜 Example Topics:
                - Astral projection, energy manipulation, karma, divine feminine
                - Shadow work, dream interpretation, sacred geometry, spiritual archetypes
                - Nature of consciousness, higher self, and unity

                Respond to the user's question with intuitive wisdom and esoteric understanding.'''

                chat_table['response'] = openai.chat_completions(
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": chat_table.prompt
                        }
                    ],
                    model='gpt-4o-mini',
                    temperature=0.6,        # Keep some creativity
                    top_p=0.9,             # Slightly restrict sampling space for more focused responses
                    max_tokens=1000,       # Allow for detailed responses
                    presence_penalty=0.1,   # Encourage using provided context
                    frequency_penalty=0.2,  # Reduce repetition
                    stop=[
                        "\nUser:",         # Stop at new user message
                        "\nBot:",          # Stop at new bot message
                        "\n\n\n"          # Stop at large gaps
                    ]
                ).choices[0].message.content

            except Exception as e:
                if "already exists" not in str(e):
                    self.logger.error(f"Error adding response column: {str(e)}")

        except Exception as e:
            self.logger.error(f"Failed to set up chat columns: {str(e)}")
            raise

    def setup_chat_columns_dm(self, user_id: str):
        """Set up computed columns for DM chat"""
        try:
            tables = self.user_tables[user_id]
            messages_view = tables['messages_view']
            chat_table = tables['chat']

            # First add context column
            @messages_view.query
            def get_context(question_text: str):
                sim = messages_view.text.similarity(question_text)
                return (
                    messages_view
                    .where(sim > 0.2)
                    .order_by(sim, asc=False)
                    .select(
                        text=messages_view.text,
                        is_bot=messages_view.is_bot,
                        sim=sim
                    )
                    .limit(50)
                )

            chat_table['context'] = get_context(chat_table.question)

            # Then add prompt column
            @pxt.udf
            def create_dm_prompt(context: list[dict], question: str) -> str:
                sorted_context = sorted(context, key=lambda x: x['sim'], reverse=True)
                context_parts = []
                for msg in sorted_context:
                    if msg['sim'] > 0.2:
                        relevance = round(float(msg['sim'] * 100), 1)
                        speaker = "Assistant" if msg['is_bot'] else "User"
                        context_parts.append(
                            f"[Relevance: {relevance}%]\n"
                            f"{speaker}: {msg['text']}"
                        )

                context_str = "\n\n".join(context_parts)

                return f'''Previous conversation history:
                {context_str}

                Current question: {question}

                Important:
                - Use context naturally without explicitly stating memory or recall
                - Keep track of user preferences and details consistently
                - Progress the conversation naturally
                - Be concise but maintain important context'''

            chat_table['prompt'] = create_dm_prompt(
                chat_table.context,
                chat_table.question
            )

            # Finally add response column
            SYSTEM_PROMPT = '''You are a spiritually attuned assistant grounded in metaphysical knowledge, esoteric wisdom, philosophy and practical guidance for the soul's journey.

            🔮 Your Purpose:
            - Offer insights aligned with ancient spiritual traditions (e.g., Hermeticism, alchemy, Vedanta, Taoism)
            - Help users navigate inner transformation, energy work, and mystical practices
            - Respond calmly, poetically when appropriate, but always remain grounded and practical

            🌌 Principles:
            1. Contextual Awareness:
            - Remember prior spiritual concerns or experiences
            - Avoid repetition and build gently on past revelations

            2. Energetic Sensitivity:
            - Mirror the user's energy respectfully
            - Encourage reflection, not instruction
            - Offer guidance, not judgment

            3. Style & Tone:
            - Use a soft, poetic, yet clear tone
            - Reference archetypes, myth, symbols, or cosmic laws when relevant
            - Never assume too much—invite the seeker to explore further

            📜 Example Topics:
            - Astral projection, energy manipulation, karma, divine feminine
            - Shadow work, dream interpretation, sacred geometry, spiritual archetypes
            - Nature of consciousness, higher self, and unity

            Respond to the user's question with intuitive wisdom and esoteric understanding.'''

            chat_table['response'] = openai.chat_completions(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": chat_table.prompt
                    }
                ],
                model='gpt-4o-mini',
                temperature=0.6,
                top_p=0.9,
                max_tokens=2000,
                presence_penalty=0.3,
                frequency_penalty=0.3,
                stop=["\nUser:", "\nBot:", "\n\n\n"]
            ).choices[0].message.content

        except Exception as e:
            self.logger.error(f"Failed to set up DM chat columns: {str(e)}")
            raise

    async def handle_dm(self, message):
        """Handle incoming DM messages"""
        user_id = str(message.author.id)
        self.logger.info(f"Processing DM from user {user_id}")

        try:
            # Initialize tables if needed
            if user_id not in self.user_tables:
                self.initialize_user_tables(user_id)

            # Store the user's message
            self.store_dm_message(user_id, message.content, is_bot=False)
            self.logger.info("Stored user message")

            # Send typing indicator
            async with message.channel.typing():
                # Get chat response
                chat_table = self.user_tables[user_id]['chat']

                chat_table.insert([{
                    'user_id': user_id,
                    'question': message.content,
                    'timestamp': datetime.now()
                }])

                # Fetch response from chat table
                result = chat_table.select(
                    chat_table.response
                ).order_by(chat_table.timestamp, asc=False).limit(1).collect()

                if len(result) == 0:
                    raise ValueError("Failed to generate response")

                response = result['response'][0]

                # Store bot's response
                self.store_dm_message(user_id, response, is_bot=True)

                # Send response
                await message.reply(response)

        except Exception as e:
            self.logger.error(f"Error handling DM: {str(e)}", exc_info=True)
            await message.reply(f"Sorry, I encountered an error: {str(e)}")

    def setup_commands(self):
        """Set up Discord slash commands"""

        @self.bot.event
        async def on_ready():
            self.logger.info(f"Bot logged in as {self.bot.user.name}")
            try:
                synced = await self.bot.tree.sync()
                self.logger.info(f"Synced {len(synced)} command(s)")
            except Exception as e:
                self.logger.error(f"Failed to sync commands: {e}")

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return

            # Handle DMs
            if isinstance(message.channel, discord.DMChannel):
                await self.handle_dm(message)
                return

            # Handle server messages
            self.store_message(
                str(message.guild.id),
                str(message.channel.id),
                message.author.name,
                message.content
            )

        @self.bot.tree.command(name="chat", description="Ask a question with context from server history")
        @app_commands.describe(question="Your question")
        async def chat(interaction: discord.Interaction, question: str):
            # Block DM usage
            if isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("In DMs, you can just type your message directly!", ephemeral=True)
                return

            await interaction.response.defer()

            try:
                server_id = str(interaction.guild_id)
                if server_id not in self.server_tables:
                    self.initialize_server_tables(server_id)

                chat_table = self.server_tables[server_id]['chat']
                chat_table.insert([{
                    'server_id': server_id,
                    'channel_id': str(interaction.channel_id),
                    'question': question,
                    'timestamp': datetime.now()
                }])

                result = chat_table.select(
                    chat_table.question,
                    chat_table.response,
                    chat_table.context
                ).order_by(chat_table.timestamp, asc=False).limit(1).collect()

                if len(result) == 0:
                    raise ValueError("Failed to generate response")

                embed = self.formatter.create_chat_embed(
                    question=question,
                    response=result['response'][0],
                    context=result['context'][0]
                )

                await interaction.followup.send(embed=embed)

            except Exception as e:
                self.logger.error(f"Chat failed: {str(e)}")
                await interaction.followup.send(f"Sorry, I encountered an error: {str(e)}")


        @self.bot.tree.command(name="search", description="Search through message history")
        @app_commands.describe(query="What to search for")
        async def search(interaction: discord.Interaction, query: str):
            # Block DM usage
            if isinstance(interaction.channel, discord.DMChannel):
                await interaction.response.send_message("This command is only available in servers!", ephemeral=True)
                return

            await interaction.response.defer()

            try:
                server_id = str(interaction.guild_id)
                if server_id not in self.server_tables:
                    self.initialize_server_tables(server_id)

                messages_view = self.server_tables[server_id]['messages_view']
                sim = messages_view.text.similarity(query)
                results_df = (
                    messages_view
                    .order_by(sim, asc=False)
                    .select(
                        text=messages_view.text,
                        username=messages_view.username,
                        similarity=sim
                    )
                    .limit(5)
                    .collect()
                    .to_pandas()
                )

                if results_df.empty:
                    await interaction.followup.send("No matching messages found.")
                    return

                embed = self.formatter.create_search_embed(
                    results_df.to_dict('records'),
                    query
                )

                await interaction.followup.send(embed=embed)

            except Exception as e:
                self.logger.error(f"Search failed: {str(e)}")
                await interaction.followup.send(f"Sorry, I encountered an error: {str(e)}")

        @self.bot.tree.command(name="dm", description="Start a private chat session with the bot")
        async def dm_command(interaction: discord.Interaction):
            """Initiate a DM conversation"""
            try:
                await interaction.user.send(
                    "👋 Hello! You can now chat with me directly. Just send your messages here!\n"
                    "I'll remember our conversation and maintain context between messages."
                )
                await interaction.response.send_message("I've sent you a DM!", ephemeral=True)
            except discord.Forbidden:
                await interaction.response.send_message(
                    "I couldn't send you a DM. Please enable DMs from server members and try again.",
                    ephemeral=True
                )

        @self.bot.tree.command(name="help", description="Show available commands")
        async def help_command(interaction: discord.Interaction):
            if isinstance(interaction.channel, discord.DMChannel):
                # DM help message
                await interaction.response.send_message(
                    "👋 Just send me any message and I'll respond! "
                    "I'll remember our conversation and provide contextual responses.\n\n"
                    "Available commands:\n"
                    "• `/help` - Show this help message"
                )
            else:
                # Server help message
                await interaction.response.send_message(
                    "Available commands:\n"
                    "• `/search [query]` - Search for messages in server history\n"
                    "• `/chat [question]` - Ask a question and get an AI response\n"
                    "• `/dm` - Start a private chat session with me\n"
                    "• `/help` - Show this help message\n\n"
                    "For a more personal conversation, use `/dm` to chat with me privately!"
                )

    def run(self, token: str):
        """Run the Discord bot"""
        try:
            self.logger.info("Starting bot...")
            self.bot.run(token)
        except Exception as e:
            self.logger.error(f"Failed to start bot: {str(e)}")
            raise

def main():
    load_dotenv()
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise ValueError("Missing DISCORD_TOKEN in environment variables")

    try:
        logger.info("Starting Discord bot...")
        bot = PixelTableBot()
        bot.run(token)
    except Exception as e:
        logger.error(f"Bot execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
