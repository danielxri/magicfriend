import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { RotateCcw, Download, Sparkles, MessageCircle } from "lucide-react";

interface ResultDisplayProps {
  originalImage: string;
  editedImage: string;
  onTryAgain: () => void;
  onUpgradeToPremium: () => void;
  isPremiumAvailable: boolean;
}

export function ResultDisplay({
  originalImage,
  editedImage,
  onTryAgain,
  onUpgradeToPremium,
  isPremiumAvailable,
}: ResultDisplayProps) {
  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = editedImage;
    link.download = "magic-friend.png";
    link.click();
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center gap-6 p-4 w-full max-w-2xl mx-auto"
    >
      <motion.h2
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-3xl font-bold text-center"
      >
        Your Magic Friend!
      </motion.h2>

      <div className="relative w-full aspect-square rounded-3xl overflow-hidden shadow-2xl animate-glow-pulse">
        <img
          src={editedImage}
          alt="Your toy with a cute face"
          className="w-full h-full object-cover"
          data-testid="img-result"
        />
        
        <motion.div
          className="absolute top-4 left-4 w-20 h-20 rounded-xl overflow-hidden border-4 border-white shadow-lg"
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <img
            src={originalImage}
            alt="Original photo"
            className="w-full h-full object-cover"
            data-testid="img-original-thumbnail"
          />
        </motion.div>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-4 w-full">
        <Button
          variant="outline"
          onClick={onTryAgain}
          data-testid="button-try-again"
          className="rounded-full px-8 py-6 text-lg font-bold"
        >
          <RotateCcw className="w-5 h-5 mr-2" />
          Try Again
        </Button>

        <Button
          onClick={handleDownload}
          data-testid="button-download"
          className="rounded-full px-8 py-6 text-lg font-bold"
        >
          <Download className="w-5 h-5 mr-2" />
          Save
        </Button>
      </div>

      {isPremiumAvailable && (
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-4 p-6 bg-gradient-to-r from-primary/10 via-secondary/10 to-accent/10 rounded-3xl border border-primary/20 w-full"
        >
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="flex-1 text-center sm:text-left">
              <h3 className="text-xl font-bold flex items-center justify-center sm:justify-start gap-2">
                <Sparkles className="w-5 h-5 text-primary" />
                Want to talk to your friend?
              </h3>
              <p className="text-muted-foreground mt-1">
                Upgrade to Premium and have real conversations!
              </p>
            </div>
            <Button
              onClick={onUpgradeToPremium}
              data-testid="button-upgrade-from-result"
              className="rounded-full px-8 py-6 text-lg font-bold bg-gradient-to-r from-primary to-accent"
            >
              <MessageCircle className="w-5 h-5 mr-2" />
              Start Talking
            </Button>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
