import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Sparkles, MessageCircle, Mic, Video, Star, X } from "lucide-react";

interface PremiumUpsellProps {
  onClose: () => void;
  onUpgrade: () => void;
}

export function PremiumUpsell({ onClose, onUpgrade }: PremiumUpsellProps) {
  const features = [
    { icon: MessageCircle, title: "Real Conversations", description: "Talk with your toy friend" },
    { icon: Mic, title: "Voice Recognition", description: "Just speak naturally" },
    { icon: Video, title: "Animated Avatar", description: "Watch them come alive" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
    >
      <motion.div
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        className="relative w-full max-w-lg"
      >
        <Card className="p-8 relative overflow-visible">
          <Button
            size="icon"
            variant="ghost"
            onClick={onClose}
            data-testid="button-close-upsell"
            className="absolute top-4 right-4 rounded-full"
          >
            <X className="w-5 h-5" />
          </Button>

          <div className="text-center mb-8">
            <motion.div
              className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-primary to-accent mb-4"
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Sparkles className="w-10 h-10 text-white" />
            </motion.div>
            <h2 className="text-3xl font-bold mb-2">Unlock the Magic!</h2>
            <p className="text-muted-foreground text-lg">
              Make your toys come alive and talk to them!
            </p>
          </div>

          <div className="space-y-4 mb-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.1 * index }}
                className="flex items-center gap-4 p-4 rounded-2xl bg-muted/50"
              >
                <div className="flex items-center justify-center w-12 h-12 rounded-full bg-primary/10">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="text-center mb-6">
            <div className="flex items-center justify-center gap-1 mb-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <Star key={i} className="w-5 h-5 fill-secondary text-secondary" />
              ))}
            </div>
            <p className="text-sm text-muted-foreground">Loved by thousands of kids!</p>
          </div>

          <Button
            onClick={onUpgrade}
            data-testid="button-upgrade-premium"
            className="w-full rounded-full py-8 text-xl font-bold bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] hover:bg-[position:100%_0]"
          >
            <Sparkles className="w-6 h-6 mr-2" />
            Try Premium Now
          </Button>

          <p className="text-center text-sm text-muted-foreground mt-4">
            Start your magical adventure today!
          </p>
        </Card>

        {[...Array(8)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-4 h-4 rounded-full"
            style={{
              background: `hsl(${280 + i * 20}, 70%, 60%)`,
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -20, 0],
              opacity: [0.5, 1, 0.5],
              scale: [0.8, 1.2, 0.8],
            }}
            transition={{
              duration: 2 + Math.random(),
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </motion.div>
    </motion.div>
  );
}
