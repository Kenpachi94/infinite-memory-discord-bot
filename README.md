python -m venv venv
venv\Scripts\activate
python bot.py


# 🤖 PixelBot: Infinite Memory Discord Assistant 
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/) 
[![Discord.py](https://img.shields.io/badge/discord.py-2.0%2B-blue.svg)](https://github.com/Rapptz/discord.py) 
[![Pixeltable](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Railway](https://img.shields.io/badge/Railway-Deployed-success)](https://railway.app/project/fdff26cf-bb1b-4fc6-ae40-76608785b337)
[![My Discord (1306431018890166272)](https://img.shields.io/badge/💬-Discord-%235865F2.svg)](https://discord.gg/QPyqFYx2UN)

A Discord bot that remembers your conversations! Built with [Pixeltable](https://github.com/pixeltable/pixeltable) and OpenAI.

## 🎮 Try It Now!
Add PixelBot to your server instantly through the [Discord Application Directory](https://discord.com/application-directory/1304932122611552346)!

1. Click "Add to Server"
2. Follow Discord's authorization flow
3. Start chatting in servers and/or DMs!

## ✨ Features
- 🧠 **Perfect Memory**: Maintains context across entire conversations
- 🔍 **Smart Search**: Find past messages based on meaning, not just keywords
- 💬 **Natural Chat**: Responds like someone who remembers your preferences
- 📱 **DM Support**: Works in both servers and private messages

## 🚀 Build Your Own

### Option 1: Deploy Your Own Instance
Deploy your own version of PixelBot using, e.g., Railway:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/project/fdff26cf-bb1b-4fc6-ae40-76608785b337)

1. Click the Deploy button above
2. Connect your GitHub account
3. Configure environment variables:
   - `DISCORD_TOKEN`
   - `OPENAI_API_KEY`
4. Deploy!

### Option 2: Local Development
```bash
# Setup
git clone https://github.com/yourusername/PixelBot.git
cd PixelBot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
# Add to .env:
DISCORD_TOKEN=your-discord-token
OPENAI_API_KEY=your-openai-key

# Run
python bot.py
```

## 💡 Commands
- `/chat [question]`: Get context-aware responses
- `/search [query]`: Find similar past messages
- `/dm`: Start private chat session

## 🛠️ Built With
- [Pixeltable](https://github.com/pixeltable/pixeltable): AI Data Infrastructure
- [Discord.py](https://github.com/Rapptz/discord.py): For Discord integration
- [OpenAI GPT-4](https://openai.com): For natural language understanding
- [Railway](https://railway.app): For deployment and hosting

## 📚 Learn More
- [Documentation](https://docs.pixeltable.com/)
- [Discord Support](https://discord.gg/QPyqFYx2UN)
- [GitHub Issues](https://github.com/pixeltable/pixeltable/issues)
- [Railway Setup Guide](https://railway.app/project/fdff26cf-bb1b-4fc6-ae40-76608785b337)
