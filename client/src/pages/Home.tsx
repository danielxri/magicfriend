import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { ThemeToggle } from "@/components/ThemeToggle";
import { ModeToggle } from "@/components/ModeToggle";
import { CameraCapture } from "@/components/CameraCapture";
import { LoadingAnimation } from "@/components/LoadingAnimation";
import { ResultDisplay } from "@/components/ResultDisplay";
import { PremiumUpsell } from "@/components/PremiumUpsell";
import { PremiumConversation } from "@/components/PremiumConversation";
import { Sparkles, Wand2 } from "lucide-react";
import type { CharacterCard } from "@shared/schema";

type AppState = "camera" | "processing" | "result" | "premium-conversation";

interface SessionData {
  sessionId: string;
  originalImageUrl: string;
  editedImageUrl: string;
  characterCard: CharacterCard | null;
}

export default function Home() {
  const [mode, setMode] = useState<"free" | "premium">("free");
  const [appState, setAppState] = useState<AppState>("camera");
  const [showUpsell, setShowUpsell] = useState(false);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);

  const generateCuteFaceMutation = useMutation({
    mutationFn: async (imageBase64: string) => {
      const response = await apiRequest("POST", "/api/generate-cute-face", { imageBase64 });
      return response.json() as Promise<SessionData>;
    },
    onSuccess: (data) => {
      setSessionData(data);
      setAppState("result");
    },
    onError: (error) => {
      console.error("Failed to generate cute face:", error);
      setAppState("camera");
    },
  });

  const handleCapture = (imageBase64: string) => {
    setAppState("processing");
    generateCuteFaceMutation.mutate(imageBase64);
  };

  const handleTryAgain = () => {
    setSessionData(null);
    setAppState("camera");
  };

  const handleModeChange = (newMode: "free" | "premium") => {
    if (newMode === "premium" && mode === "free") {
      if (sessionData?.editedImageUrl) {
        setAppState("premium-conversation");
        setMode("premium");
      } else {
        setShowUpsell(true);
      }
    } else {
      setMode(newMode);
      if (newMode === "free" && appState === "premium-conversation") {
        setAppState(sessionData?.editedImageUrl ? "result" : "camera");
      }
    }
  };

  const handleUpgradeToPremium = () => {
    setShowUpsell(false);
    setMode("premium");
    if (sessionData?.editedImageUrl) {
      setAppState("premium-conversation");
    }
  };

  const handleBackFromConversation = () => {
    setMode("free");
    setAppState(sessionData?.editedImageUrl ? "result" : "camera");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-muted/30 flex flex-col">
      <header className="sticky top-0 z-50 w-full p-4 flex items-center justify-between gap-4 bg-background/80 backdrop-blur-sm border-b border-border">
        <motion.div
          className="flex items-center gap-2"
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
        >
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center">
            <Wand2 className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-bold hidden sm:block">Magic Friends</h1>
        </motion.div>

        <ModeToggle mode={mode} onModeChange={handleModeChange} />

        <ThemeToggle />
      </header>

      <main className="flex-1 flex flex-col items-center justify-center p-4 relative overflow-hidden">
        <AnimatePresence mode="wait">
          {appState === "camera" && (
            <motion.div
              key="camera"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full max-w-4xl flex flex-col items-center gap-6"
            >
              <div className="text-center mb-4">
                <motion.h2
                  className="text-3xl sm:text-4xl font-bold mb-2"
                  initial={{ y: -10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.1 }}
                >
                  {mode === "free" ? "Take a Photo!" : "Capture Your Friend!"}
                </motion.h2>
                <motion.p
                  className="text-lg text-muted-foreground"
                  initial={{ y: -10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  {mode === "free"
                    ? "Point at something and watch it get a cute face!"
                    : "Take a photo and start a magical conversation!"}
                </motion.p>
              </div>

              <CameraCapture
                onCapture={handleCapture}
                isProcessing={generateCuteFaceMutation.isPending}
              />

              {mode === "premium" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2 text-primary"
                >
                  <Sparkles className="w-5 h-5" />
                  <span className="font-medium">Premium Mode Active</span>
                </motion.div>
              )}
            </motion.div>
          )}

          {appState === "processing" && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="w-full flex items-center justify-center"
            >
              <LoadingAnimation
                message={
                  mode === "premium"
                    ? "Creating your magical friend..."
                    : "Adding a cute face..."
                }
              />
            </motion.div>
          )}

          {appState === "result" && sessionData && (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="w-full"
            >
              <ResultDisplay
                originalImage={sessionData.originalImageUrl}
                editedImage={sessionData.editedImageUrl}
                onTryAgain={handleTryAgain}
                onUpgradeToPremium={() => handleModeChange("premium")}
                isPremiumAvailable={mode === "free"}
              />
            </motion.div>
          )}

          {appState === "premium-conversation" && sessionData && (
            <motion.div
              key="conversation"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full h-[calc(100vh-5rem)]"
            >
              <PremiumConversation
                sessionId={sessionData.sessionId}
                editedImageUrl={sessionData.editedImageUrl}
                characterCard={sessionData.characterCard}
                onBack={handleBackFromConversation}
              />
            </motion.div>
          )}
        </AnimatePresence>

        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 rounded-full bg-primary/20"
              style={{
                left: `${10 + i * 15}%`,
                top: `${20 + (i % 3) * 25}%`,
              }}
              animate={{
                y: [0, -30, 0],
                opacity: [0.2, 0.5, 0.2],
              }}
              transition={{
                duration: 3 + i * 0.5,
                repeat: Infinity,
                delay: i * 0.5,
              }}
            />
          ))}
        </div>
      </main>

      <AnimatePresence>
        {showUpsell && (
          <PremiumUpsell
            onClose={() => setShowUpsell(false)}
            onUpgrade={handleUpgradeToPremium}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
