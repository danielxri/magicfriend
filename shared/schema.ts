import { sql } from "drizzle-orm";
import { pgTable, text, varchar, serial, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Users table (simple for now)
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Sessions table for storing character cards and conversation state
export const sessions = pgTable("sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  originalImageUrl: text("original_image_url"),
  editedImageUrl: text("edited_image_url"),
  didSourceUrl: text("did_source_url"),
  talkingPhotoId: text("talking_photo_id"),
  characterCard: jsonb("character_card").$type<CharacterCard>(),
  conversationHistory: jsonb("conversation_history").$type<ConversationMessage[]>().default([]),
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
});

export const insertSessionSchema = createInsertSchema(sessions).omit({
  id: true,
  createdAt: true,
});

export type InsertSession = z.infer<typeof insertSessionSchema>;
export type Session = typeof sessions.$inferSelect;

// Character card type for personality matching
export interface CharacterCard {
  name: string;
  personality: string;
  voiceStyle: string;
  backstory: string;
  speakingPatterns: string[];
  favoriteThings: string[];
}

// Conversation message type
export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

// API request/response types
export const generateCuteFaceSchema = z.object({
  imageBase64: z.string(),
});

export const createCharacterCardSchema = z.object({
  sessionId: z.string(),
  imageDescription: z.string().optional(),
});

export const sendMessageSchema = z.object({
  sessionId: z.string(),
  message: z.string(),
});

export type GenerateCuteFaceRequest = z.infer<typeof generateCuteFaceSchema>;
export type CreateCharacterCardRequest = z.infer<typeof createCharacterCardSchema>;
export type SendMessageRequest = z.infer<typeof sendMessageSchema>;
