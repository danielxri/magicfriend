import { motion } from "framer-motion";
import { Sparkles, Camera } from "lucide-react";

interface ModeToggleProps {
  mode: "free" | "premium";
  onModeChange: (mode: "free" | "premium") => void;
}

export function ModeToggle({ mode, onModeChange }: ModeToggleProps) {
  return (
    <div className="relative flex items-center bg-card rounded-full p-1 shadow-lg border border-border">
      <motion.div
        className="absolute top-1 bottom-1 bg-primary rounded-full"
        initial={false}
        animate={{
          left: mode === "free" ? "4px" : "calc(50% + 2px)",
          width: "calc(50% - 6px)",
        }}
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      />
      
      <button
        onClick={() => onModeChange("free")}
        data-testid="button-mode-free"
        className={`relative z-10 flex items-center gap-2 px-6 py-3 rounded-full font-semibold transition-colors ${
          mode === "free" ? "text-primary-foreground" : "text-muted-foreground"
        }`}
      >
        <Camera className="w-5 h-5" />
        <span>Free</span>
      </button>
      
      <button
        onClick={() => onModeChange("premium")}
        data-testid="button-mode-premium"
        className={`relative z-10 flex items-center gap-2 px-6 py-3 rounded-full font-semibold transition-colors ${
          mode === "premium" ? "text-primary-foreground" : "text-muted-foreground"
        }`}
      >
        <Sparkles className="w-5 h-5" />
        <span>Premium</span>
      </button>
    </div>
  );
}
