import { motion } from "framer-motion";
import { Sparkles, Star, Heart, Wand2 } from "lucide-react";

interface LoadingAnimationProps {
  message?: string;
}

export function LoadingAnimation({ message = "Making magic..." }: LoadingAnimationProps) {
  const icons = [Sparkles, Star, Heart, Wand2];
  
  return (
    <div className="flex flex-col items-center justify-center gap-8 p-8">
      <div className="relative w-32 h-32">
        {icons.map((Icon, index) => (
          <motion.div
            key={index}
            className="absolute inset-0 flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.5, rotate: 0 }}
            animate={{
              opacity: [0, 1, 1, 0],
              scale: [0.5, 1, 1, 0.5],
              rotate: [0, 0, 0, 45],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: index * 0.5,
              times: [0, 0.2, 0.8, 1],
            }}
          >
            <Icon
              className="w-16 h-16"
              style={{
                color: `hsl(${280 + index * 40}, 70%, 55%)`,
              }}
            />
          </motion.div>
        ))}
        
        <motion.div
          className="absolute inset-0 rounded-full border-4 border-primary/30"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute inset-4 rounded-full border-4 border-secondary/50"
          animate={{ rotate: -360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />
      </div>

      <motion.p
        className="text-2xl font-bold text-foreground"
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        {message}
      </motion.p>

      <div className="flex gap-2">
        {[0, 1, 2, 3, 4].map((i) => (
          <motion.div
            key={i}
            className="w-3 h-3 rounded-full bg-primary"
            animate={{ y: [0, -12, 0] }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: i * 0.1,
            }}
          />
        ))}
      </div>
    </div>
  );
}
