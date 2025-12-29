import { motion } from "framer-motion";
import type { CharacterCard } from "@shared/schema";

interface AvatarDisplayProps {
  imageUrl: string;
  characterCard?: CharacterCard | null;
  isSpeaking: boolean;
  videoRef?: React.RefObject<HTMLVideoElement>;
}

export function AvatarDisplay({
  imageUrl,
  characterCard,
  isSpeaking,
  videoRef,
}: AvatarDisplayProps) {
  return (
    <div className="relative w-full max-w-md mx-auto">
      <motion.div
        className={`relative aspect-square rounded-3xl overflow-hidden shadow-2xl ${
          isSpeaking ? "animate-glow-pulse" : ""
        }`}
        animate={isSpeaking ? { scale: [1, 1.02, 1] } : { scale: 1 }}
        transition={{ duration: 0.5, repeat: isSpeaking ? Infinity : 0 }}
      >
        {videoRef ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-full object-cover"
            data-testid="video-avatar"
          />
        ) : (
          <img
            src={imageUrl}
            alt={characterCard?.name || "Your magical friend"}
            className="w-full h-full object-cover"
            data-testid="img-avatar"
          />
        )}

        {isSpeaking && (
          <motion.div
            className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {[0, 1, 2, 3, 4].map((i) => (
              <motion.div
                key={i}
                className="w-2 h-2 rounded-full bg-primary"
                animate={{ scaleY: [1, 2, 1] }}
                transition={{
                  duration: 0.4,
                  repeat: Infinity,
                  delay: i * 0.1,
                }}
              />
            ))}
          </motion.div>
        )}
      </motion.div>

      {characterCard && (
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="mt-4 text-center"
        >
          <h3 className="text-2xl font-bold" data-testid="text-character-name">
            {characterCard.name}
          </h3>
          <p className="text-muted-foreground" data-testid="text-character-personality">
            {characterCard.personality}
          </p>
        </motion.div>
      )}
    </div>
  );
}
