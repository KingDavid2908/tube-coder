// src/app/page.tsx
"use client";

import React, { useState, useCallback, useEffect } from "react";
import { ChatProvider, useChat } from "@/contexts/ChatContext";
import InputArea from "@/components/InputArea";
import ChatWindow from "@/components/ChatWindow";
import { TrashIcon } from "@heroicons/react/24/outline";
import { extractYouTubeVideoId } from "@/utils/youtubeHelper";
import { v4 as uuidv4 } from 'uuid'; 

function ChatInterface() {
  const {
    addMessage,
    updateMessageContent,
    sessionId, 
    setSessionId, 
    startNewSession,
    isProcessing,
    setIsProcessing,
    messages
  } = useChat();

  const handleSendMessage = useCallback(
    async (messageText: string) => {
      const trimmedInput = messageText.trim();
      if (!trimmedInput || isProcessing) return;

      console.log("Handling message:", trimmedInput);
      
      let videoId: string | null = null;
      const urlRegex = /(https?:\/\/[^\s<>"]+|www\.[^\s<>"]+)/;
      const urlMatch = trimmedInput.match(urlRegex);
      if (urlMatch && urlMatch[0]) {
        const foundUrl = urlMatch[0];
        videoId = extractYouTubeVideoId(foundUrl);
        if (videoId) {
            console.log("Extracted YouTube Video ID:", videoId);
        } else {
             console.log("Found a URL, but could not extract a valid YouTube ID from it:", foundUrl);
        }
      } else {
          console.log("No URL found in message.");
      }
      const isNewVideoTask = !!videoId; 

      let sessionIdForRequest: string;
      if (isNewVideoTask) {
          sessionIdForRequest = uuidv4();
          console.log(`Generated NEW sessionId for this request: ${sessionIdForRequest}`);
          setSessionId(sessionIdForRequest);
      } else {
          sessionIdForRequest = sessionId;
          console.log(`Reusing existing sessionId for this request: ${sessionIdForRequest}`);
      }
      
      setIsProcessing(true);

      addMessage({
        role: "user",
        content: trimmedInput,
        videoId: videoId ?? undefined,
      });

      const assistantMessage = addMessage({
        role: "assistant",
        content: "",
        isLoading: true,
      });

      try {
        const apiUrl = process.env.NEXT_PUBLIC_BACKEND_CHAT_API_URL || "http://localhost:8080/api/chat";
        const apiPayload = {
          message: trimmedInput,
          sessionId: sessionIdForRequest,
        };
        console.log(`[handleSendMessage] Sending API request with sessionId: ${sessionIdForRequest}`, apiPayload);
        const response = await fetch(apiUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(apiPayload),
        });

        if (response.ok) {
          const data = await response.json();
          console.log("[handleSendMessage] Received OK response, data:", data);
          updateMessageContent(assistantMessage.id, data.reply);

          if (data.session_id && sessionIdForRequest !== data.session_id) {
             console.warn(`[handleSendMessage] Backend session ID (${data.session_id}) differs from frontend's used ID (${sessionIdForRequest}). Trusting frontend.`);
          }

        } else {
            let errorMsg = `Request failed with status ${response.status}`;
            try {
              const errorData = await response.json();
              errorMsg = `Error: ${errorData.detail || errorData.error || JSON.stringify(errorData)}`;
            } catch (e) {}
            console.error("API Error:", errorMsg);
            updateMessageContent(assistantMessage.id, errorMsg);
        }
      } catch (error) {
         console.error("Network or Fetch Error:", error);
         let errorMsg = "Error: Could not connect to the backend.";
         if (error instanceof Error) errorMsg = `Network Error: ${error.message}`;
         updateMessageContent(assistantMessage.id, errorMsg);

      } finally {
        setIsProcessing(false);
      }
    },
    [addMessage, sessionId, setSessionId, updateMessageContent, isProcessing, setIsProcessing]
  );
  const handleStopProcessing = useCallback(async () => {
    console.log(`[handleStopProcessing] Attempting to stop processing for session: ${sessionId}`);
    setIsProcessing(false);
    const loadingMessage = messages.findLast(m => m.role === 'assistant' && m.isLoading);

    if (loadingMessage) {
      console.log(`[handleStopProcessing] Found loading message to update: ${loadingMessage.id}`);
      updateMessageContent(loadingMessage.id, "[Processing stopped by user]");
    } else {
      console.log("[handleStopProcessing] No active loading assistant message found to update.");
    }

    try {
      const cancelUrl = process.env.NEXT_PUBLIC_BACKEND_CANCEL_API_URL || "http://localhost:8080/api/cancel_request";
      console.log(`[handleStopProcessing] Sending cancellation request to: ${cancelUrl} for session ${sessionId}`);
      await fetch(cancelUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }), 
      });
    } catch (error) {
      console.error("[handleStopProcessing] Error sending cancellation request:", error);
    }
  }, [sessionId, setIsProcessing, updateMessageContent, messages]);

  const handleClearChat = () => {
    console.log("Clearing chat history and starting new session...");
    startNewSession();
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
       {/* Header */}
       <header className="text-cyan-400 p-3 shadow-md flex-shrink-0 flex items-center justify-between bg-radial from-purple-800 via-indigo-950 to-black">
         {/* Left Side - Logo */}
         <div className="flex items-center justify-start w-1/3">
           <img src="/logo.png" alt="Vectronix Logo" className="h-12 w-auto max-w-48" />
         </div>
         {/* Centered Title */}
         <div className="flex items-center justify-center w-1/3">
           <h1 className="text-lg font-semibold text-center whitespace-nowrap">TubeCoder Agent</h1>
         </div>
         {/* Right Side - Clear Button */}
         <div className="flex items-center justify-end w-1/3">
           <button onClick={handleClearChat} title="Clear Chat History & Start New Session" className="p-1.5 text-red-500 hover:text-red-400 hover:bg-white hover:bg-opacity-10 rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-red-500 transition-colors" aria-label="Clear chat history">
             <TrashIcon className="h-5 w-5" />
           </button>
         </div>
       </header>
       <ChatWindow />
       <footer className="p-3 bg-gray-100 border-t border-gray-300 flex-shrink-0">
         <InputArea onSendMessage={handleSendMessage} isProcessing={isProcessing} onStopProcessing={handleStopProcessing} />
       </footer>
     </div>
  );
}

export default function HomePage() {
  return (
    <ChatProvider>
      <ChatInterface />
    </ChatProvider>
  );
}