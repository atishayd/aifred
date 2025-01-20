# AiFRED

AiFRED is a Python-based application designed to manage and analyze student engagement, course attendance, and question relevance within an educational environment. 

## Features
- Face recognition for attendance.
- Hand-raise detection using Mediapipe.
- Relevance analysis of questions with OpenAI APIs.
- Stereo audio recording, transcription via Whisper, and GPT-based classification.
- MongoDB for data storage.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/atishayd/aifred.git
   ```

2. Navigate into the project directory:
   ```bash
   cd aifred
   ```

3. (Optional) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Linux/macOS)
   # or
   venv\Scripts\activate     # (Windows)
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

AiFRED relies on settings from:
- A local environment file (e.g., `.env`) or environment variables
- The `config.json` file for certain default values

To set up your configuration:

1. Copy `config.template.json` to `config.json`:
   ```bash
   cp config.template.json config.json
   ```

2. Edit `config.json` with your specific settings:
   - MongoDB URI
   - Database name
   - Any custom thresholds or settings

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   MONGODB_URI=your_mongodb_uri
   ```
