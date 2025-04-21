// src/contexts/ChatContext.tsx
"use client";

import React, { createContext, useState, useContext, ReactNode, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

// Define ChatMessage interface
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  isLoading?: boolean;
  videoId?: string;
}

// Define ChatContextProps interface
interface ChatContextProps {
  sessionId: string;
  setSessionId: React.Dispatch<React.SetStateAction<string>>;
  messages: ChatMessage[];
  addMessage: (message: Omit<ChatMessage, 'id'>) => ChatMessage;
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  startNewSession: () => void;
  updateMessageContent: (id: string, newContent: string) => void;
  setMessageLoading: (id: string, isLoading: boolean) => void;
  isProcessing: boolean;
  setIsProcessing: React.Dispatch<React.SetStateAction<boolean>>;
}

// --- localStorage Keys (Add a version suffix in case structure changes later) ---
const SESSION_ID_STORAGE_KEY = 'tubeCoder_sessionId_v1';
const MESSAGES_STORAGE_KEY = 'tubeCoder_messages_v1';

const ChatContext = createContext<ChatContextProps>(null!);

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

export const ChatProvider = ({ children }: { children: ReactNode }) => {
  // --- Initialize state with defaults FIRST ---
  const [sessionId, setSessionId] = useState<string>(() => uuidv4());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // --- Add hydration state ---
  const [isHydrated, setIsHydrated] = useState(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  // --- Effect to load from localStorage AFTER initial mount ---
  useEffect(() => {
    try {
      const storedSessionId = localStorage.getItem(SESSION_ID_STORAGE_KEY);
      const storedMessages = localStorage.getItem(MESSAGES_STORAGE_KEY);

      if (storedSessionId) {
        setSessionId(storedSessionId);
        console.log("Hydrated sessionId from localStorage:", storedSessionId);
      } else {
        localStorage.setItem(SESSION_ID_STORAGE_KEY, sessionId);
      }

      if (storedMessages) {
        const parsedMessages = JSON.parse(storedMessages);
        setMessages(parsedMessages.map((msg: any) => ({ ...msg, isLoading: false })));
        console.log("Hydrated messages from localStorage");
      }
    } catch (error) {
      console.error("Error hydrating state from localStorage:", error);
      localStorage.removeItem(MESSAGES_STORAGE_KEY);
      localStorage.removeItem(SESSION_ID_STORAGE_KEY);
      setSessionId(uuidv4());
      setMessages([]);
    } finally {
      setIsHydrated(true);
    }
  }, []);

  // --- Effect to SAVE state to localStorage on change ---
  useEffect(() => {
    // Only save *after* initial hydration is complete AND on client side
    if (isHydrated && typeof window !== 'undefined') {
      console.log("Saving sessionId to localStorage:", sessionId);
      localStorage.setItem(SESSION_ID_STORAGE_KEY, sessionId);

      const messagesToSave = messages.map(({ isLoading, ...rest }) => rest);
      try {
        localStorage.setItem(MESSAGES_STORAGE_KEY, JSON.stringify(messagesToSave));
        console.log(`Saved ${messagesToSave.length} messages to localStorage.`);
      } catch (error) {
        console.error("Error saving messages to localStorage:", error);
      }
    }
  }, [sessionId, messages, isHydrated]);

  // --- Functions ---

  const startNewSession = useCallback(() => {
    const newId = uuidv4();
    setMessages([]);
    setSessionId(newId);
    setIsProcessing(false);
    console.log("Started new chat session:", newId);
  }, [setSessionId, setMessages, setIsProcessing]);

  const addMessage = useCallback((messageData: Omit<ChatMessage, 'id'>): ChatMessage => {
    const newMessage: ChatMessage = {
      ...messageData,
      id: uuidv4(),
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    console.log("Added message:", newMessage.id, "Role:", newMessage.role);
    return newMessage;
  }, [setMessages]);

  const updateMessageContent = useCallback((id: string, newContent: string) => {
    setMessages((prevMessages) =>
      prevMessages.map((msg) =>
        msg.id === id ? { ...msg, content: newContent, isLoading: false } : msg
      )
    );
    console.log("Updated message content:", id);
  }, [setMessages]);

  const setMessageLoading = useCallback((id: string, isLoading: boolean) => {
    setMessages((prevMessages) =>
      prevMessages.map((msg) =>
        msg.id === id ? { ...msg, isLoading: isLoading } : msg
      )
    );
    console.log("Set message loading:", id, "to", isLoading);
  }, [setMessages]);

  const value = {
    sessionId,
    setSessionId,
    messages,
    addMessage,
    setMessages,
    startNewSession,
    updateMessageContent,
    setMessageLoading,
    isProcessing,
    setIsProcessing,
  };

  return (
    <ChatContext.Provider value={value}>
      {isHydrated ? children : null}
    </ChatContext.Provider>
  );
};