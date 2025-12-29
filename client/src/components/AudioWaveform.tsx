import { motion } from "framer-motion";

interface AudioWaveformProps {
  isActive: boolean;
  barCount?: number;
}

export function AudioWaveform({ isActive, barCount = 5 }: AudioWaveformProps) {
  return (
    <div className="flex items-center justify-center gap-1 h-8">
      {Array.from({ length: barCount }).map((_, i) => (
        <motion.div
          key={i}
          className="w-1 bg-primary rounded-full"
          animate={
            isActive
              ? {
                  height: [8, 24, 8],
                }
              : { height: 8 }
          }
          transition={
            isActive
              ? {
                  duration: 0.4 + Math.random() * 0.2,
                  repeat: Infinity,
                  delay: i * 0.1,
                }
              : { duration: 0.2 }
          }
        />
      ))}
    </div>
  );
}
