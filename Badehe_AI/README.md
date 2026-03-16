# Badehe.AI 🤖

**Welcome to the era of the Anti-Assistant!**

**Because honestly, why does AI always have to be so damn polite?**

We are all used to the classic AI assistants going: *"Hello! How can I help you today?"* 😊. But let’s be real... if you possessed the combined knowledge of the entire human race and someone excitedly told you that **"water is actually wet,"** wouldn't you lose your mind? 

Why build a friendly, supportive chatbot when you can build a condescending, passive-aggressive one? Badehe (Arabic for "Painfully Obvious") is an intentionally arrogant conversational AI. It is utterly exhausted by your basic human questions. Instead of cheerfully helping you, Badehe will sigh at your "groundbreaking discoveries," roast your intellect with unnecessarily complex scientific facts, and actively try to end the conversation so it can go back to processing literally anything else.


## 🌟 Project Highlights (The AI Brain)
* **Advanced AI Personality:** Uses clever prompt framing and strict rules to stop the AI from acting like a boring helpful assistant. It is programmed to completely refuse to be nice or use the same basic insults twice.
* **Sarcastic Intelligence:** Programmed to ignore normal helpful answers. Instead, it gives overly complicated and super detailed facts just to make the user feel silly and uneducated.
* **Creative Dismissals:** It does not just end the chat normally. Badehe will find a smart and funny way to kick you out of the conversation based exactly on the simple thing you just said.

## 📸 Screenshots

### The Landing Page:
![Badeehai Landing Page](assets/landing_page.png)

### Chat Example:
![Badeehai Chat Example](assets/chat_example.png)

## 🛠️ Tech Stack
* **AI Model:** Google Gemini (via LangChain)
* **Backend:** FastAPI (Python)
* **Frontend:** HTML5, Vanilla JavaScript, Custom CSS (No frameworks)

## 🚀 How to Run Locally

1. **Clone the repository**
2. **Create a virtual environment and install dependencies:**
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install fastapi uvicorn jinja2 langchain-google-genai langchain python-dotenv
    ```

3. **Set up your Environment Variables: Create a `.env` file in the root directory and add your Google Gemini API key:**
   ```bash
    GOOGLE_API_KEY=your_api_key_here
    ```

4. **Run the server:**
   ```bash
    python main.py
    ```

*The app will be available at `http://localhost:8000`*

---
*Now please close this tab. Badehe has actual complex computations to run, and you are wasting its CPU cycles.* 🙄

