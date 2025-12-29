import type { User, InsertUser, Session, CharacterCard, ConversationMessage } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Session management
  createSession(originalImageUrl?: string, editedImageUrl?: string): Promise<Session>;
  getSession(id: string): Promise<Session | undefined>;
  updateSession(id: string, updates: Partial<Session>): Promise<Session | undefined>;
  updateSessionCharacterCard(id: string, characterCard: CharacterCard): Promise<Session | undefined>;
  addConversationMessage(id: string, message: ConversationMessage): Promise<Session | undefined>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private sessions: Map<string, Session>;

  constructor() {
    this.users = new Map();
    this.sessions = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async createSession(originalImageUrl?: string, editedImageUrl?: string): Promise<Session> {
    const id = randomUUID();
    const session: Session = {
      id,
      originalImageUrl: originalImageUrl || null,
      editedImageUrl: editedImageUrl || null,
      didSourceUrl: null,
      talkingPhotoId: null,
      characterCard: null,
      conversationHistory: [],
      createdAt: new Date(),
    };
    this.sessions.set(id, session);
    return session;
  }

  async getSession(id: string): Promise<Session | undefined> {
    return this.sessions.get(id);
  }

  async updateSession(id: string, updates: Partial<Session>): Promise<Session | undefined> {
    const session = this.sessions.get(id);
    if (!session) return undefined;
    
    const updated = { ...session, ...updates };
    this.sessions.set(id, updated);
    return updated;
  }

  async updateSessionCharacterCard(id: string, characterCard: CharacterCard): Promise<Session | undefined> {
    return this.updateSession(id, { characterCard });
  }

  async addConversationMessage(id: string, message: ConversationMessage): Promise<Session | undefined> {
    const session = this.sessions.get(id);
    if (!session) return undefined;
    
    const conversationHistory = [...(session.conversationHistory || []), message];
    return this.updateSession(id, { conversationHistory });
  }
}

export const storage = new MemStorage();
