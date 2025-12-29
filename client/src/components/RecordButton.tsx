import { motion } from "framer-motion";
import { Mic, MicOff, Square } from "lucide-react";

interface RecordButtonProps {
  isRecording: boolean;
  isProcessing: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;
}

export function RecordButton({
  isRecording,
  isProcessing,
  onStartRecording,
  onStopRecording,
}: RecordButtonProps) {
  const handleClick = () => {
    if (isRecording) {
      onStopRecording();
    } else {
      onStartRecording();
    }
  };

  return (
    <div className="relative flex items-center justify-center">
      {isRecording && (
        <>
          <motion.div
            className="absolute w-32 h-32 rounded-full bg-destructive/20"
            animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.2, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          <motion.div
            className="absolute w-40 h-40 rounded-full bg-destructive/10"
            animate={{ scale: [1, 1.3, 1], opacity: [0.3, 0.1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
          />
        </>
      )}

      <motion.button
        onClick={handleClick}
        disabled={isProcessing}
        data-testid="button-record"
        className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-colors ${
          isRecording
            ? "bg-destructive"
            : isProcessing
            ? "bg-muted"
            : "bg-primary"
        }`}
        whileTap={{ scale: 0.95 }}
      >
        {isProcessing ? (
          <div className="w-8 h-8 border-4 border-foreground/30 border-t-foreground rounded-full animate-spin" />
        ) : isRecording ? (
          <Square className="w-10 h-10 text-destructive-foreground fill-current" />
        ) : (
          <Mic className="w-10 h-10 text-primary-foreground" />
        )}
      </motion.button>

      <p className="absolute -bottom-8 text-sm font-medium text-muted-foreground whitespace-nowrap">
        {isProcessing
          ? "Thinking..."
          : isRecording
          ? "Tap to stop"
          : "Tap to talk"}
      </p>
    </div>
  );
}
