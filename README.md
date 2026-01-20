# Agentic Memory AI Bot

A simple terminal chatbot that uses OpenAI for responses and Mem0 for conversational memory. Conversation history is stored and retrieved per user so the assistant can recall prior details (e.g., your name).

## Prerequisites
- Python 3.11+
- An OpenAI API key
- A Mem0 API key (from https://mem0.ai)

## Setup
1) Clone or open this folder
2) Create and activate a virtual environment (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3) Install dependencies inside the venv:
   ```powershell
   pip install openai python-dotenv mem0ai
   ```
4) Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=sk-...
   MEM0_API_KEY=m0-...
   ```

## Run
```powershell
python basicChatBot.py
```
Then type messages at the `User:` prompt. The bot will:
- Look up related memories for your `user_id`
- Feed those memories into the system prompt
- Answer with OpenAI `gpt-4o`
- Save the latest exchange back to Mem0

## How memory works
- `user_id` is hardcoded in the script as `"avb"`. Change it to separate users.
- Searches include a filter `{ "user_id": user_id }` to scope memories per user.
- When no memories exist, the bot will continue without them.

## Troubleshooting
- **400 from Mem0 search (filters required):** Ensure the call includes `filters={"user_id": user_id}` (already in the script).
- **Module not found (mem0):** Confirm you installed `mem0ai` inside the activated `.venv` and are importing `from mem0 import MemoryClient`.
- **Auth errors:** Recheck keys in `.env`, then restart the shell so `load_dotenv()` picks them up.

## Files
- basicChatBot.py — main chat loop with OpenAI + Mem0 integration
- .env — stores API keys (ignored by git)
- .gitignore — excludes `.env`, `.venv/`, bytecode

## Notes
- The script uses `gpt-4o`; you can switch to another available model in `basicChatBot.py`.
- Keep your API keys secret; never commit `.env`.
