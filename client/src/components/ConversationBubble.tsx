import { motion } from "framer-motion";
import type { ConversationMessage } from "@shared/schema";

interface ConversationBubbleProps {
  message: ConversationMessage;
  characterName?: string;
}

export function ConversationBubble({ message, characterName = "Friend" }: ConversationBubbleProps) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}
    >
      <div
        className={`max-w-[85%] px-5 py-3 rounded-3xl ${
          isUser
            ? "bg-primary text-primary-foreground rounded-br-lg"
            : "bg-card border border-border rounded-bl-lg"
        }`}
        data-testid={`bubble-${message.role}-${message.timestamp}`}
      >
        {!isUser && (
          <p className="text-xs font-semibold text-primary mb-1">{characterName}</p>
        )}
        <p className="text-base leading-relaxed">{message.content}</p>
      </div>
    </motion.div>
  );
}
