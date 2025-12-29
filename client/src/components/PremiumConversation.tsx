import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { ArrowLeft, Volume2, VolumeX, Send, Loader2 } from "lucide-react";
import { RecordButton } from "./RecordButton";
import { ConversationBubble } from "./ConversationBubble";
import { AudioWaveform } from "./AudioWaveform";
import { LoadingAnimation } from "./LoadingAnimation";
import type { CharacterCard, ConversationMessage } from "@shared/schema";

interface PremiumConversationProps {
  sessionId: string;
  editedImageUrl: string;
  characterCard: CharacterCard | null;
  onBack: () => void;
}

export function PremiumConversation({
  sessionId,
  editedImageUrl,
  characterCard,
  onBack,
}: PremiumConversationProps) {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentCharacter, setCurrentCharacter] = useState<CharacterCard | null>(characterCard);
  const [textInput, setTextInput] = useState("");
  const [talkingPhotoId, setTalkingPhotoId] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<string>("Setting up your friend...");
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string | null>(null);
  const [showPlayOverlay, setShowPlayOverlay] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const uploadTalkingPhoto = useCallback(async () => {
    // No-op for local MuseTalk integration
    // The image is already stored in the session, and we generate video on demand.
    console.log("[MuseTalk] Ready for local generation");
    setConnectionStatus("Ready to chat!");
    setIsLoading(false);
    setTalkingPhotoId("local-mode");
  }, []);

  const generateAndPlayVideo = useCallback(async (text: string) => {
    setIsGeneratingVideo(true);
    setConnectionStatus("Creating video response...");

    try {
      console.log("[MuseTalk] Generating video for text:", text.substring(0, 50) + "...");

      const response = await fetch("/api/generate-avatar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          text,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("[MuseTalk] Video generation request failed:", errorData);
        throw new Error(errorData.error || "Failed to generate video");
      }

      const data = await response.json();
      const videoUrl = data.videoUrl;

      if (videoUrl) {
        console.log("[MuseTalk] Video ready:", videoUrl);
        setCurrentVideoUrl(videoUrl);
        setIsSpeaking(true);
        setConnectionStatus("Ready to chat!");
      } else {
        throw new Error("No video URL returned");
      }
    } catch (error) {
      console.error("[MuseTalk] Video generation failed:", error);
      setConnectionStatus("Video generation failed. Using audio only.");
      speakText(text); // Fallback to TTS
    } finally {
      setIsGeneratingVideo(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (!characterCard) {
      generateCharacterCard();
    } else {
      uploadTalkingPhoto();
    }
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (currentVideoUrl && videoRef.current) {
      console.log("[HeyGen] Playing video:", currentVideoUrl);
      videoRef.current.src = currentVideoUrl;
      videoRef.current.load();

      const playPromise = videoRef.current.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            console.log("Autoplay started");
            setShowPlayOverlay(false);
          })
          .catch(e => {
            console.error("[HeyGen] Autoplay prevented:", e);
            setIsSpeaking(false);
            setShowPlayOverlay(true);
          });
      }
    }
  }, [currentVideoUrl]);

  const handleManualPlay = () => {
    if (videoRef.current && currentVideoUrl) {
      videoRef.current.play();
      setIsSpeaking(true);
      setShowPlayOverlay(false);
    }
  };

  const handleVideoEnded = () => {
    console.log("[HeyGen] Video playback ended");
    setIsSpeaking(false);
    setCurrentVideoUrl(null);
  };

  const generateCharacterCard = async () => {
    try {
      setConnectionStatus("Creating character...");
      const response = await fetch("/api/character-card", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId }),
      });
      const data = await response.json();
      setCurrentCharacter(data.characterCard);
      uploadTalkingPhoto();
    } catch (error) {
      console.error("Failed to generate character card:", error);
      setIsLoading(false);
    }
  };

  const speakText = (text: string) => {
    console.log("[TTS] Starting speakText with:", text.substring(0, 50) + "...");

    if (!("speechSynthesis" in window)) {
      console.warn("[TTS] Speech synthesis not supported");
      setIsSpeaking(false);
      return;
    }

    window.speechSynthesis.cancel();
    setIsSpeaking(true);

    const speak = () => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1.2;

      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(
        (v) => v.name.includes("Female") || v.name.includes("Samantha") || v.name.includes("Karen")
      );
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    };

    if (window.speechSynthesis.getVoices().length === 0) {
      window.speechSynthesis.onvoiceschanged = () => speak();
    } else {
      speak();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Failed to start recording:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob);
      formData.append("sessionId", sessionId);

      const response = await fetch("/api/conversation/message", {
        method: "POST",
        body: formData,
      });

      const reader = response.body?.getReader();
      if (!reader) return;

      let assistantMessage = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        const lines = text.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "transcript") {
                setMessages((prev) => [
                  ...prev,
                  { role: "user", content: data.text, timestamp: Date.now() },
                ]);
              } else if (data.type === "response_chunk") {
                assistantMessage += data.text;
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage?.role === "assistant") {
                    lastMessage.content = assistantMessage;
                  } else {
                    newMessages.push({
                      role: "assistant",
                      content: assistantMessage,
                      timestamp: Date.now(),
                    });
                  }
                  return newMessages;
                });
              } else if (data.type === "done") {
                setIsProcessing(false);
                if (!isMuted && assistantMessage) {
                  await generateAndPlayVideo(assistantMessage);
                }
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("Failed to process audio:", error);
      setIsProcessing(false);
    }
  };

  const processText = async (text: string) => {
    if (!text.trim()) return;

    console.log("[ProcessText] Starting with text:", text);
    setIsProcessing(true);
    setTextInput("");

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text, timestamp: Date.now() },
    ]);

    try {
      const response = await fetch("/api/conversation/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, message: text }),
      });

      const reader = response.body?.getReader();
      if (!reader) return;

      let assistantMessage = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          setIsProcessing(false);
          if (!isMuted && assistantMessage) {
            await generateAndPlayVideo(assistantMessage);
          }
          break;
        }

        const textData = new TextDecoder().decode(value);
        const lines = textData.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "response_chunk") {
                assistantMessage += data.text;
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage?.role === "assistant") {
                    lastMessage.content = assistantMessage;
                  } else {
                    newMessages.push({
                      role: "assistant",
                      content: assistantMessage,
                      timestamp: Date.now(),
                    });
                  }
                  return newMessages;
                });
              } else if (data.type === "done") {
                setIsProcessing(false);
              } else if (data.type === "error") {
                console.error("[ProcessText] Received error from server:", data.message);
                setIsProcessing(false);
                setConnectionStatus("Error: " + data.message);
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("[ProcessText] Failed:", error);
      setIsProcessing(false);
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (textInput.trim() && !isProcessing && !isGeneratingVideo) {
      processText(textInput);
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <LoadingAnimation message={connectionStatus} />
      </div>
    );
  }

  return (
    <div className="flex flex-col lg:flex-row h-full gap-4 p-4 overflow-auto">
      <div className="lg:flex-1 flex flex-col items-center justify-center p-4 relative shrink-0">
        <div className="absolute top-4 left-4 z-10">
          <Button
            size="icon"
            variant="ghost"
            onClick={onBack}
            data-testid="button-back"
            className="rounded-full"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </div>

        <div className="absolute top-4 right-4 z-10">
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setIsMuted(!isMuted)}
            data-testid="button-mute"
            className="rounded-full"
          >
            {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
          </Button>
        </div>

        <div className="relative">
          <motion.div
            className="relative cursor-pointer"
            onClick={showPlayOverlay ? handleManualPlay : undefined}
            animate={isSpeaking ? { scale: [1, 1.02, 1] } : {}}
            transition={{ duration: 0.5, repeat: isSpeaking ? Infinity : 0 }}
          >
            {currentVideoUrl ? (
              <video
                ref={videoRef}
                poster={editedImageUrl}
                autoPlay
                playsInline
                muted={isMuted}
                onEnded={handleVideoEnded}
                className="w-64 h-64 sm:w-80 sm:h-80 lg:w-[512px] lg:h-[512px] rounded-full object-cover border-4 border-purple-400 shadow-lg"
                data-testid="video-avatar"
              />
            ) : (
              <img
                src={editedImageUrl}
                alt={currentCharacter?.name || "Your friend"}
                className="w-64 h-64 sm:w-80 sm:h-80 lg:w-[512px] lg:h-[512px] rounded-full object-cover border-4 border-purple-400 shadow-lg"
                data-testid="img-avatar"
              />
            )}
            {isSpeaking && (
              <motion.div
                className="absolute -inset-2 rounded-full border-4 border-purple-400"
                animate={{ scale: [1, 1.05, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              />
            )}

            {showPlayOverlay && currentVideoUrl && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-full z-20">
                <div className="bg-white/90 rounded-full p-2 shadow-lg">
                  <Volume2 className="w-8 h-8 text-primary fill-current" />
                </div>
              </div>
            )}

          </motion.div>

          {isGeneratingVideo && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-full z-20">
              <Loader2 className="w-8 h-8 animate-spin text-white" />
            </div>
          )}
        </div>

        <div className="mt-4 text-center">
          <h2 className="text-2xl font-bold" data-testid="text-character-name">
            {currentCharacter?.name || "Your Friend"}
          </h2>
          <p className="text-muted-foreground text-sm">
            {currentCharacter?.personality?.substring(0, 50)}...
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {connectionStatus}
          </p>
        </div>

        <AnimatePresence>
          {(isRecording || isSpeaking) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="mt-6"
            >
              <AudioWaveform isActive={isRecording || isSpeaking} barCount={7} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="flex-1 flex flex-col bg-background rounded-3xl border border-border overflow-hidden min-h-[300px]">
        <div className="p-4 border-b border-border">
          <h3 className="font-bold text-lg" data-testid="text-conversation-title">
            Chat with {currentCharacter?.name || "your friend"}
          </h3>
        </div>

        <ScrollArea className="flex-1 p-4" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-8">
              <motion.p
                className="text-muted-foreground text-lg"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                Type a message or tap the microphone to say hello to {currentCharacter?.name || "your friend"}!
              </motion.p>
            </div>
          ) : (
            messages.map((message, index) => (
              <ConversationBubble
                key={`${message.role}-${message.timestamp}-${index}`}
                message={message}
                characterName={currentCharacter?.name}
              />
            ))
          )}
        </ScrollArea>

        <div className="p-4 border-t border-border">
          <form onSubmit={handleTextSubmit} className="flex gap-2 mb-4">
            <Input
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Type a message..."
              disabled={isProcessing || isRecording || isGeneratingVideo}
              data-testid="input-message"
              className="flex-1"
            />
            <Button
              type="submit"
              size="icon"
              disabled={!textInput.trim() || isProcessing || isRecording || isGeneratingVideo}
              data-testid="button-send"
            >
              {isGeneratingVideo ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
          </form>
          <div className="flex justify-center">
            <RecordButton
              isRecording={isRecording}
              isProcessing={isProcessing || isGeneratingVideo}
              onStartRecording={startRecording}
              onStopRecording={stopRecording}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
