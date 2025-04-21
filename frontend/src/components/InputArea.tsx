// src/components/InputArea.tsx
"use client";

import React, { useState } from 'react';
import { PaperAirplaneIcon, StopCircleIcon } from '@heroicons/react/24/outline';

// Define the props the component expects
interface InputAreaProps {
  onSendMessage: (messageText: string) => void;
  isProcessing: boolean; // Added per suggestion
  onStopProcessing: () => void; // Added per suggestion
}

export default function InputArea({ onSendMessage, isProcessing, onStopProcessing }: InputAreaProps) {
  const [input, setInput] = useState(""); // State to hold the text input value

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault(); // Prevent default page reload on form submission
    // Only submit if not processing (Send button is visible)
    if (!isProcessing) {
      const trimmedInput = input.trim();
      if (!trimmedInput) return;
      onSendMessage(trimmedInput);
      setInput(""); // Clear input only on successful send
    }
  };

  const handleStopClick = () => {
    onStopProcessing(); // Call the stop handler from props
    // DO NOT clear the input here
  };

  // Common button classes (adjust colors if needed, but user asked for same)
  const buttonBaseClasses = "p-2 ml-2 text-gray-500 rounded-full hover:bg-blue-100 hover:text-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-gray-500 transition-colors duration-150";

  return (
    // The main form element
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      {/* Container for the input and button */}
      <div className="flex items-center bg-white shadow rounded-lg p-2 border border-gray-200">
        {/* Text Input */}
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isProcessing ? "Processing..." : "Enter YouTube URL or ask a question..."} // Modified per suggestion
          className="flex-grow p-2 bg-transparent focus:outline-none text-sm text-gray-800 placeholder-gray-500 disabled:bg-gray-100" // Modified per suggestion
          autoFocus
          disabled={isProcessing} // Added per suggestion
        />

        {/* Conditional Button Rendering */}
        {isProcessing ? (
          // Stop Button
          <button
            type="button" // Important: Not type="submit"
            onClick={handleStopClick}
            className={buttonBaseClasses} // Use same base style
            aria-label="Stop processing"
            title="Stop current request" // Tooltip
          >
            <StopCircleIcon className="w-5 h-5" />
          </button>
        ) : (
          // Send Button
          <button
            type="submit" // This makes Enter key work
            disabled={!input.trim()} // Keep disabled logic for empty input
            className={buttonBaseClasses}
            aria-label="Send message"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        )}
      </div>
    </form>
  );
}