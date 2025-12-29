# Imagine Friend 2 - System Specification v1.0

## 1. Upgrade Context
**Purpose**: This document captures the exact functional state of the "Imagine Friend 2" web application as of Dec 2025. It serves as the "Golden Reference" for the upcoming iOS Native App port, ensuring no critical logic or optimizations are lost.

## 2. High-Level Architecture
The system consists of three primary components:
1.  **Backend A (Node.js)**: Orchestrator, Database, Image Generation (DALL-E), Character Logic (LLM).
2.  **Backend B (Python AI Service)**: High-Performance GPU Avatar Video Generation (MuseTalk).
3.  **Frontend (React/Vite)**: User Interface, Audio Recording, Video Playback.

---

## 3. Core Functionalities

### A. "Magical Object" Creation (Image Transformation)
*   **Trigger**: User uploads/takes a photo of an object.
*   **Preprocessing (Node.js)**:
    *   Image is resized to 1024x1024.
    *   **Transparency Mask**: A circular mask of **radius 300px** is applied to the center (alpha=0). This tells DALL-E where to generate the face.
*   **Generation (OpenAI)**:
    *   **Endpoint**: `v1/images/edits`
    *   **Model**: `gpt-image-1.5`
    *   **Prompt Key Elements**:
        *   "Isolate the object on a clean, solid background." (Crucial for mobile UI).
        *   "Face Integration": Face must appear carved/molded from material.
        *   "Expression": "Big bright eyes inspired by Pixar", "Mouth clearly visible and well-formed" (Required for Lip Sync).
        *   "Style": Photorealistic Magical Realism.
*   **Output**: A transparent PNG (or solid BG PNG) returned to UI.

### B. Personality Generation (Context-Aware)
*   **Trigger**: After image generation.
*   **Logic (Node.js)**:
    *   **Endpoint**: `/api/character-card`
    *   **Model**: **GPT-5.2 Vision** (Multimodal).
    *   **Input**: The *Original* user photo (via `image_url`).
    *   **System Prompt**: "ANALYZE THE IMAGE... IDENTIFY the object... DESIGN a personality linked to its function."
    *   **Output**: JSON with Name, Backstory, Voice Style (e.g., "Sassy Stapler").

### C. Avatar Video Generation (The "Talking" Feature)
*   **Trigger**: User sends text or audio.
*   **AI Service (Python/MuseTalk)**:
    *   **Model Architecture**:
        *   **Teacher Model (UNet)**: Standard MuseTalk UNet (SD 1.5 based). *Note: Student model was tested but rejected due to quality issues.*
        *   **VAE**: `AutoencoderKL` (FP16).
        *   **Whisper**: `whisper-tiny` (FP16).
        *   **Face Parsing**: BiSeNet validation (90px cheek width).
    *   **Critical Optimizations (Turbo)**:
        *   **Batch Size**: **48** (Maximized for Blackwell GB10).
        *   **FPS**: **21** (Matches Playback Safety Buffer).
        *   **XFormers**: Enabled (`xformers.ops.memory_efficient_attention`).
        *   **Blending**: Pure **NumPy** implementation (No PIL) + **Safe Crop** (Padding for out-of-bounds faces).
    *   **Streaming Pipeline**:
        *   FFmpeg acts as a pipe (`-f image2pipe`).
        *   **Threaded Reader**: Prevents stdout deadlocks.
        *   **Low Latency Flags**: `-tune zerolatency`, `-frag_duration 100000` (100ms), `-g 21` (1s GOP).

### D. Conversation Interface
*   **Frontend**:
    *   **Avatar Size**: **512px x 512px** (approx 2x original size).
    *   **Audio**: Uses `MediaRecorder` API (WebM) -> Transcribed by OpenAI Whisper (Node.js).
    *   **Polling/Stream**:
        *   Text Chat: Server-Sent Events (SSE).
        *   Video: Streamed via Proxy (`/api/proxy/stream/:id` -> Python `:8001`).

---

## 4. Critical "Gotchas" (Do Not Regression)
1.  **Buffering Loop**: Playback FPS (Browser) MUST be <= Generation FPS. We use **21 FPS** for playback. If you go higher (25/30), the buffer empties, player stalls, and loops.
2.  **Face Detection**: The generic "Pixar" prompt sometimes makes eyes/mouths too abstract. The **300px Mask** is the sweet spot. A 400px mask allowed too much "body" generation, breaking face detection.
3.  **Proxying**: iOS requires HTTPS/Secure streams often. The Node.js proxy handles headers. Direct Python access often fails CORS or Mixed Content.
4.  **Audio Latency**: FFmpeg *must* have audio mapped explicitly (`-map 0:v -map 1:a`) and fragmented (`-movflags frag_keyframe...`) or iOS AVPlayer will wait for the whole file.

## 5. Technical Stack Summary
| Component | Technology | Config |
| :--- | :--- | :--- |
| **Node.js** | TypeScript, Express | Port 5001 |
| **Database** | PostgreSQL | Session Store |
| **AI Service** | Python, PyTorch, CUDA | Port 8001 |
| **GPU** | NVIDIA Blackwell (GB10) | Single GPU |
| **Models** | MuseTalk (Teacher), GPT-image-1.5, GPT-5.2 | |
