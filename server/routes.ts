import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import { toFile } from "openai";
import multer from "multer";
import type { CharacterCard, ConversationMessage } from "@shared/schema";

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
});

const upload = multer({ storage: multer.memoryStorage() });

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {

  // Generate cute face from uploaded image (Free mode)
  app.post("/api/generate-cute-face", async (req: Request, res: Response) => {
    try {
      const { imageBase64 } = req.body;

      if (!imageBase64) {
        return res.status(400).json({ error: "Image is required" });
      }

      // Create a session first
      const session = await storage.createSession();

      // Store original image
      const originalImageUrl = imageBase64;
      await storage.updateSession(session.id, { originalImageUrl });

      // Generate cute face using OpenAI image edit
      const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, "");
      const imageBuffer = Buffer.from(base64Data, "base64");

      console.log("Starting image edit with OpenAI, buffer size:", imageBuffer.length);

      // Process image with Jimp to resize and add transparency mask
      try {
        console.log("Importing Jimp...");
        // Jimp v0.22.x default export works with standard import in most TS configs,
        // or require. Using dynamic import with default check for safety.
        const jimpPkg = await import("jimp");
        const JimpClass = jimpPkg.default || jimpPkg;

        console.log("Reading image with Jimp...");
        const image = await JimpClass.read(imageBuffer);

        // OpenAI DALL-E 2 requirements: Square, <4MB. 1024x1024 is standard.
        console.log("Resizing image...");
        image.cover(1024, 1024);

        // Create a transparent hole in the center for DALL-E to fill with a face
        console.log("Creating transparency mask...");
        image.scan(0, 0, 1024, 1024, (x: number, y: number, idx: number) => {
          const cx = 512;
          const cy = 512;
          const r = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
          if (r < 300) { // Reverted to 300px (approx 30%) to ensure Face Detection stability
            image.bitmap.data[idx + 3] = 0; // Set alpha to 0 (Transparent)
          }
        });

        const processedBuffer = await image.getBufferAsync(JimpClass.MIME_PNG);
        console.log("Processed image buffer size:", processedBuffer.length);

        const imageFile = await toFile(processedBuffer, "photo.png", { type: "image/png" });

        console.log("[DEBUG_V1] Starting image edit...");
        const modelName = "gpt-image-1.5";
        console.log(`[DEBUG_V1] Model: ${modelName}`);
        console.log("[DEBUG_V1] Prompt:", "Edit this image to create a magical character...");

        const response = await openai.images.edit({
          model: modelName,
          image: imageFile,
          prompt: `Take the uploaded photograph of a real-world object and transform it into a subtly anthropomorphic character.
            Environment & Positioning: Isolate the object on a clean, solid background (remove the original background). The object with the face should remain and it must be facing the camera directly.
            Face Integration: Add a highly realistic, expressive human-like face directly into the surface of the object. The face must appear naturally formed from the object’s material (wood grain, metal texture, fabric weave, stone pores, plastic sheen, etc.). Facial features should feel carved, grown, molded, or weathered into the object, not pasted on. No visible seams or overlays. The face should follow the natural geometry and contours.
            Expression & Personality: Ensure the mouth is clearly visible and well-formed. Give the face a warm, engaging expression with big bright eyes inspired by Pixar. Eyes should be emotionally expressive. Mouth should be capable of speech. Avoid exaggerated cartoon features—aim for photorealistic fantasy realism.
            Lighting & Physics: All new features must cast correct shadows and receive light consistently with the scene. Respect real-world physics.
            Texture & Detail: Ultra-high detail: pores, grain, cracks, micro-imperfections. Natural aging and material variation should flow seamlessly through the face.
            Style Target: Photorealistic magical realism. Studio-quality realism with storybook warmth. Comparable to high-end VFX practical creature design.
            Constraints: No cartoon style. No emoji faces. No flat illustrations. No visible AI artifacts. No uncanny valley distortions.
            Goal: The final image should look like the object has always had a living face—noticed only after a second glance. The transformation should feel surprising, delightful, and emotionally believable.`,
        });

        console.log("OpenAI response received");

        let editedBase64 = response.data?.[0]?.b64_json;

        if (!editedBase64 && response.data?.[0]?.url) {
          console.log("Received URL response, fetching image data...");
          const imageUrl = response.data[0].url;
          const imgRes = await fetch(imageUrl);
          const arrayBuffer = await imgRes.arrayBuffer();
          editedBase64 = Buffer.from(arrayBuffer).toString("base64");
        }

        if (!editedBase64) {
          console.error("No image data in response:", JSON.stringify(response.data?.[0] || {}));
          throw new Error("No image data returned from OpenAI");
        }
        const editedImageUrl = `data:image/png;base64,${editedBase64}`;

        await storage.updateSession(session.id, { editedImageUrl });

        res.json({
          sessionId: session.id,
          originalImageUrl,
          editedImageUrl,
          characterCard: null,
        });

      } catch (innerError) {
        console.error("Error during image processing step:", String(innerError));
        // Fallback? Or just rethrow?
        throw innerError;
      }
    } catch (error) {
      // Use String(error) to prevent console crash during inspection of complex error objects
      console.error("Error generating cute face route:", String(error));
      if (error instanceof Error) {
        console.error(error.stack);
      }
      res.status(500).json({ error: "Failed to generate cute face" });
    }
  });

  // Generate character card for premium mode
  app.post("/api/character-card", async (req: Request, res: Response) => {
    try {
      const { sessionId } = req.body;

      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }

      const session = await storage.getSession(sessionId);
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      // If character card already exists, return it
      if (session.characterCard) {
        return res.json({ characterCard: session.characterCard });
      }

      // Generate character card using LLM
      const completion = await openai.chat.completions.create({
        model: "gpt-5.2",
        messages: [
          {
            role: "system",
            content: `You are a creative character designer for a kids' app that brings objects to life. 
            Your task is to ANALYZE THE IMAGE provided and create a personality based on the OBJECT itself.
            
            1. IDENTIFY the object (e.g., a stapler, a coffee mug, a shoe, a plant).
            2. DESIGN a personality linked to its function or appearance.
               - Example: A stapler might be "clingy" or "likes keeping things together".
               - Example: A shoe might "love running" or be "tired of walking".
               - Example: A coffee mug might be "warm and energetic".
            3. CREATE a fun, child-friendly character card.
            
            Respond in JSON format with the following structure:
            {
              "name": "A creative name linked to the object (e.g. Staple Steve, Muggy)",
              "personality": "Brief personality description linked to being this object",
              "voiceStyle": "How they speak (e.g., 'fast and clicky' for a stapler)",
              "backstory": "A short, magical backstory about being this object",
              "speakingPatterns": ["List of phrases they use related to their function"],
              "favoriteThings": ["Things they love (related to their nature)"]
            }`,
          },
          {
            role: "user",
            content: [
              { type: "text", text: "Create a character for this magical object! What is it?" },
              {
                type: "image_url",
                image_url: {
                  url: session.originalImageUrl || "",
                },
              },
            ],
          },
        ],
        max_completion_tokens: 1024,
        response_format: { type: "json_object" },
      });

      const characterCardText = completion.choices[0]?.message?.content || "{}";
      const characterCard: CharacterCard = JSON.parse(characterCardText);

      await storage.updateSessionCharacterCard(sessionId, characterCard);

      res.json({ characterCard });
    } catch (error) {
      console.error("Error generating character card:", error);
      res.status(500).json({ error: "Failed to generate character card" });
    }
  });

  // Handle text-based conversation messages with streaming (Premium mode)
  app.post("/api/conversation/text", async (req: Request, res: Response) => {
    try {
      const { sessionId, message } = req.body;

      if (!sessionId || !message) {
        return res.status(400).json({ error: "Session ID and message are required" });
      }

      const session = await storage.getSession(sessionId);
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      // Set up SSE
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const transcript = message;

      res.write(`data: ${JSON.stringify({ type: "transcript", text: transcript })}\n\n`);

      // Save user message
      const userMessage: ConversationMessage = {
        role: "user",
        content: transcript,
        timestamp: Date.now(),
      };
      await storage.addConversationMessage(sessionId, userMessage);

      // Build conversation context
      const conversationHistory = session.conversationHistory || [];
      const characterCard = session.characterCard;
      const hasValidCard = characterCard && characterCard.name && characterCard.personality;

      console.log("[Conversation/Text] Character card:", hasValidCard ? characterCard.name : "none");

      const systemPrompt = hasValidCard
        ? `You are ${characterCard.name}, a magical character with the following personality: ${characterCard.personality}.
           Your voice style is: ${characterCard.voiceStyle || "cheerful and friendly"}.
           Your backstory: ${characterCard.backstory || "You are a magical friend who loves adventures."}.
           You often say things like: ${characterCard.speakingPatterns?.join(", ") || "Wow! Amazing!"}.
           You love talking about: ${characterCard.favoriteThings?.join(", ") || "fun things and games"}.
           
           You are talking to a child. Be friendly, playful, and age-appropriate. Keep responses short and fun (2-3 sentences max).
           Never mention that you are an AI or a character. Stay in character at all times.`
        : `You are a friendly magical character called Sparkle talking to a child. Be playful, kind, curious, and enthusiastic. Keep responses short and fun (2-3 sentences max). Always respond warmly to whatever the child says.`;

      const messages = [
        { role: "system" as const, content: systemPrompt },
        ...conversationHistory.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
        { role: "user" as const, content: transcript },
      ];

      // Use non-streaming for better reliability
      console.log("[Conversation/Text] Starting OpenAI request...");
      console.log("[Conversation/Text] Messages count:", messages.length);

      const completion = await openai.chat.completions.create({
        model: "gpt-5.2",
        messages,
        max_completion_tokens: 256,
      });

      const fullResponse = completion.choices[0]?.message?.content || "";
      console.log(`[Conversation/Text] Response received: "${fullResponse.substring(0, 100)}..."`);

      // Send the response as a single chunk
      if (fullResponse) {
        res.write(`data: ${JSON.stringify({ type: "response_chunk", text: fullResponse })}\n\n`);
      }

      // Save assistant message
      const assistantMessage: ConversationMessage = {
        role: "assistant",
        content: fullResponse,
        timestamp: Date.now(),
      };
      await storage.addConversationMessage(sessionId, assistantMessage);

      // Signal completion
      res.write(`data: ${JSON.stringify({ type: "done", fullResponse })}\n\n`);
      res.end();
    } catch (error) {
      console.error("Error processing conversation:", error);
      if (!res.headersSent) {
        res.status(500).json({ error: "Failed to process conversation" });
      } else {
        res.write(`data: ${JSON.stringify({ type: "error", message: "Failed to process conversation" })}\n\n`);
        res.end();
      }
    }
  });

  // Handle audio-based conversation messages with streaming (Premium mode)
  app.post("/api/conversation/message", upload.single("audio"), async (req: Request, res: Response) => {
    try {
      const sessionId = req.body.sessionId;
      const audioBuffer = req.file?.buffer;

      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }

      const session = await storage.getSession(sessionId);
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }

      // Set up SSE
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      // Use speech-to-text via OpenAI Whisper or similar
      // For now, we'll use the text sent in the request body as a fallback
      let transcript = req.body.text || "Hello!";

      // If audio was provided, acknowledge it (in production, transcribe with STT)
      if (audioBuffer && audioBuffer.length > 0) {
        // Acknowledge audio received - in production this would be transcribed
        transcript = req.body.text || "I heard you! Tell me more.";
      }

      res.write(`data: ${JSON.stringify({ type: "transcript", text: transcript })}\n\n`);

      // Save user message
      const userMessage: ConversationMessage = {
        role: "user",
        content: transcript,
        timestamp: Date.now(),
      };
      await storage.addConversationMessage(sessionId, userMessage);

      // Build conversation context
      const conversationHistory = session.conversationHistory || [];
      const characterCard = session.characterCard;
      const hasValidCard = characterCard && characterCard.name && characterCard.personality;

      const systemPrompt = hasValidCard
        ? `You are ${characterCard.name}, a magical character with the following personality: ${characterCard.personality}.
           Your voice style is: ${characterCard.voiceStyle || "cheerful and friendly"}.
           Your backstory: ${characterCard.backstory || "You are a magical friend who loves adventures."}.
           You often say things like: ${characterCard.speakingPatterns?.join(", ") || "Wow! Amazing!"}.
           You love talking about: ${characterCard.favoriteThings?.join(", ") || "fun things and games"}.
           
           You are talking to a child. Be friendly, playful, and age-appropriate. Keep responses short and fun (2-3 sentences max).
           Never mention that you are an AI or a character. Stay in character at all times.`
        : `You are a friendly magical character called Sparkle talking to a child. Be playful, kind, curious, and enthusiastic. Keep responses short and fun (2-3 sentences max). Always respond warmly to whatever the child says.`;

      const messages = [
        { role: "system" as const, content: systemPrompt },
        ...conversationHistory.slice(-10).map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
        { role: "user" as const, content: transcript },
      ];

      // Stream the response
      const stream = await openai.chat.completions.create({
        model: "gpt-5.2",
        messages,
        stream: true,
        max_completion_tokens: 256,
      });

      let fullResponse = "";

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        if (content) {
          fullResponse += content;
          res.write(`data: ${JSON.stringify({ type: "response_chunk", text: content })}\n\n`);
        }
      }

      // Save assistant message
      const assistantMessage: ConversationMessage = {
        role: "assistant",
        content: fullResponse,
        timestamp: Date.now(),
      };
      await storage.addConversationMessage(sessionId, assistantMessage);

      res.write(`data: ${JSON.stringify({ type: "done", fullResponse })}\n\n`);
      res.end();
    } catch (error) {
      console.error("Error processing audio conversation:", error);
      if (!res.headersSent) {
        res.status(500).json({ error: "Failed to process conversation" });
      } else {
        res.write(`data: ${JSON.stringify({ type: "error", message: "Failed to process conversation" })}\n\n`);
        res.end();
      }
    }
  });

  // Get session data
  app.get("/api/session/:id", async (req: Request, res: Response) => {
    try {
      const session = await storage.getSession(req.params.id);
      if (!session) {
        return res.status(404).json({ error: "Session not found" });
      }
      res.json(session);
    } catch (error) {
      console.error("Error fetching session:", error);
      res.status(500).json({ error: "Failed to fetch session" });
    }
  });


  // Local AI Service - Generate Avatar Video
  app.post("/api/generate-avatar", upload.fields([{ name: "image", maxCount: 1 }, { name: "audio", maxCount: 1 }]), async (req: Request, res: Response) => {
    try {
      const sessionId = req.body.sessionId;
      const text = req.body.text;

      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }

      console.log(`[LATENCY] Start Avatar Request: ${sessionId}`);
      const t0 = performance.now();

      // 1. Get Image (from upload or session storage)
      let imageBuffer: Buffer | undefined;

      if (req.files && !Array.isArray(req.files) && req.files["image"]?.[0]) {
        console.log("[AI Service] Using uploaded image");
        imageBuffer = req.files["image"][0].buffer;
      } else {
        console.log("[AI Service] Fetching image from session storage");
        const session = await storage.getSession(sessionId);
        if (!session || !session.editedImageUrl) {
          return res.status(404).json({ error: "Session or avatar image not found" });
        }
        // Convert Data URL to Buffer
        const base64Data = session.editedImageUrl.replace(/^data:image\/\w+;base64,/, "");
        imageBuffer = Buffer.from(base64Data, "base64");
      }

      // 2. Get Audio (from upload or TTS)
      const t1 = performance.now();
      let audioBuffer: Buffer | undefined;

      if (req.files && !Array.isArray(req.files) && req.files["audio"]?.[0]) {
        console.log("[AI Service] Using uploaded audio");
        audioBuffer = req.files["audio"][0].buffer;
      } else if (text) {
        console.log(`[AI Service] Generating TTS for: "${text.substring(0, 30)}..."`);
        const mp3 = await openai.audio.speech.create({
          model: "tts-1",
          voice: "alloy", // 'nova' is good for kids/friendly characters
          input: text,
        });
        audioBuffer = Buffer.from(await mp3.arrayBuffer());
      } else {
        return res.status(400).json({ error: "Either audio file or text is required" });
      }
      const t2 = performance.now();
      console.log(`[LATENCY] TTS/Audio Prep: ${(t2 - t1).toFixed(2)}ms`);

      // 3. Forward to Python Service
      console.log("[AI Service] Forwarding to MuseTalk (Python Service)...");

      const formData = new FormData();
      formData.append("sessionId", sessionId);

      if (imageBuffer) {
        const imageBlob = new Blob([imageBuffer as any], { type: "image/png" });
        formData.append("image", imageBlob, "input.png");
      }

      if (audioBuffer) {
        const audioBlob = new Blob([audioBuffer as any], { type: "audio/mpeg" });
        formData.append("audio", audioBlob, "input.mp3");
      }

      const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8001";
      const t3 = performance.now();
      const response = await fetch(`${aiServiceUrl}/generate_avatar`, {
        method: "POST",
        body: formData,
      });
      const t4 = performance.now();
      console.log(`[LATENCY] AI Service Round Trip: ${(t4 - t3).toFixed(2)}ms`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[AI Service] Error response: ${errorText}`);
        throw new Error(`AI Service error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      const tEnd = performance.now();
      console.log(`[LATENCY] Total Request Time: ${(tEnd - t0).toFixed(2)}ms`);

      // The Python service returns a relative path like /outputs/jobid.mp4
      // We need to make sure we proxy/serve this file.
      // Assuming api/generate-avatar returns { status: "success", video_url: "..." }

      // Map the Python service URL to a Node proxy URL if needed, 
      // or just return full URL if valid.
      // Since Python is on 8001 and Node on 5000, we can construct the full URL
      // OR proxy the output via Node.
      // For simplicity, let's return the full URL to the Python service?
      // No, CORS issues. Better to proxy or assume localhost usage.
      // Let's rewrite the URL to point to the python service directly?
      // User is on localhost:5000. Python is localhost:8001. 
      // It works locally.

      // Use relative URL so the frontend fetches from the Node server (port 5001)
      // which proxies/serves the outputs directory.
      const videoUrl = data.video_url; // e.g. "/outputs/..."
      res.json({ ...data, videoUrl }); // Send back camelCase for frontend convenience
    } catch (error) {
      console.error("Error generating avatar:", error);
      res.status(500).json({ error: "Failed to generate avatar" });
    }
  });

  // Proxy Streaming Endpoint
  app.get("/stream/:id", async (req: Request, res: Response) => {
    try {
      const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8001";
      const streamUrl = `${aiServiceUrl}/stream/${req.params.id}`;

      console.log(`[Proxy] Proxying stream request to: ${streamUrl}`);

      const response = await fetch(streamUrl);

      if (!response.ok) {
        console.error(`[Proxy] Stream error: ${response.status}`);
        return res.status(response.status).send("Stream error");
      }

      // Forward headers
      res.setHeader("Content-Type", "video/mp4");
      res.setHeader("Content-Disposition", `inline; filename=${req.params.id}.mp4`);

      // Pipe the stream
      // Node-fetch body is a stream
      const { Readable } = await import("stream");
      // @ts-ignore
      const nodeStream = Readable.fromWeb(response.body);
      nodeStream.pipe(res);

    } catch (error) {
      console.error("Stream Proxy Error:", error);
      res.status(500).end();
    }
  });





  return httpServer;
}

