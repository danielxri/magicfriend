# Imagine Friend 2 🪄✨

**Bring everyday objects to life with AI.**

Imagine Friend 2 is a full-stack application that transforms photos of inanimate objects into animated, talking characters with unique personalities.

## 🌟 Features

*   **Magical Object Creation**: Turns a photo of an object (e.g., a stapler) into a character with a realistic face using `GPT-image-1.5`.
*   **Object-Aware Personality**: Uses **GPT-5.2 Vision** to analyze the object and generate a matching personality (e.g., "Staple Steve" who loves holding things together).
*   **Real-Time Avatar Video**: Generates lip-synced videos of the character talking in real-time using a custom-optimized **MuseTalk** pipeline (Turbo Mode: 21 FPS on GB10).
*   **Interactive Chat**: Talk to your new friend via text or voice.

## 📚 Documentation

Detailed system architecture and specifications for developers (and the iOS Port team) can be found in the `docs/` folder:

*   [System Specification v1.0](docs/system_specification_v1.md) - **Start Here** (Architecture, API, Critical Logic).
*   [Walkthrough & Optimizations](docs/walkthrough.md) - Deep dive into the performance tuning and debugging journey.

## 🛠️ Quick Start

### Prerequisites
*   Node.js (v20+)
*   Python 3.10+ (with CUDA support for AI Service)
*   Docker (Optional, for containerized AI service)
*   NVIDIA GPU (Recommended for real-time video)

### 1. Backend (Node.js)
```bash
npm install
# Set up .env with OPENAI_API_KEY, DATABASE_URL, etc.
npm run dev
# Runs on Port 5001
```

### 2. AI Service (Python)
```bash
cd ai_service
pip install -r requirements.txt
python app.py
# Runs on Port 8001
```

## ⚠️ Critical Notes for iOS Port
If you are porting the frontend to iOS, please read the **"Critical Gotchas"** section in [System Specification](docs/system_specification_v1.md).
*   **Playback FPS**: Must be capped at **21 FPS** to prevent buffering loops.
*   **Mask Size**: Keep the transparency mask at **300px** for reliable face detection.
