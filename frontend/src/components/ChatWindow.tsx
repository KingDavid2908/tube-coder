// src/components/ChatWindow.tsx
"use client";

import React, { useEffect, useRef } from 'react';
import { useChat, ChatMessage } from '@/contexts/ChatContext';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ChatWindow() {
    const { messages } = useChat();
    const messagesEndRef = useRef<HTMLDivElement>(null);
    // Function to scroll to the bottom of the chat
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // Scroll to bottom whenever messages array changes
    useEffect(() => {
        scrollToBottom();
    }, [messages]);


    return (
        <div className="flex-1 overflow-y-auto p-4 space-y-4"> {/* Space between messages */}
            {messages.map((message) => {
                let isDownloadTrigger = false;
                let sessionIdForDownload: string | null = null;
                const triggerStartText = "[DOWNLOAD_PROJECT:";
                const triggerEndText = "]";
                let displayContent = message.content;
                let showButton = false; 

                if (message.role === 'assistant' && message.content) { // Check assistant and non-empty content
                    // --- Use includes() for detection ---
                    if (message.content.includes(triggerStartText)) {
                        const startIndex = message.content.indexOf(triggerStartText);
                        const endIndex = message.content.indexOf(triggerEndText, startIndex);

                        // Check if both markers were found correctly
                        if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
                            isDownloadTrigger = true;
                            // Extract the potential session ID substring
                            const extractedId = message.content.substring(startIndex + triggerStartText.length, endIndex).trim();

                            // --- Validate extracted ID format (simple check) ---
                            if (extractedId && /^[a-zA-Z0-9_\-]+$/.test(extractedId)) {
                                sessionIdForDownload = extractedId;
                                console.log(`[ChatWindow] Trigger DETECTED! Extracted Session ID: ${sessionIdForDownload}`);

                                // Clean the trigger text from the display content
                                const triggerFullText = message.content.substring(startIndex, endIndex + triggerEndText.length);
                                displayContent = message.content.replace(triggerFullText, "").trim();
                                // Provide default text if nothing else was left
                                displayContent = displayContent || "Project is ready for download.";
                            } else {
                                // ID extraction failed or format invalid
                                console.warn(`[ChatWindow] Found trigger markers but failed to extract valid Session ID. Extracted: "${extractedId}"`);
                                isDownloadTrigger = false;
                            }
                        } else {
                            // Markers not found correctly
                            console.log(`[ChatWindow] Trigger start found, but end marker ']' missing or misplaced.`);
                            isDownloadTrigger = false;
                        }
                    }

                    // Final condition check for the button
                    showButton = isDownloadTrigger && !!sessionIdForDownload && !message.isLoading;
                }

                return (
                    <div key={message.id} className={`flex flex-col ${message.role === 'user' ? 'items-end' : 'items-start'}`}> {/* Use flex-col on outer div */}
                        {/* Message Bubble */}
                        <div className={`max-w-xl lg:max-w-2xl px-4 py-2 rounded-lg shadow flex flex-col ${ message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
                            {/* Loading Indicator or Markdown Content */}
                            {message.isLoading && message.role === 'assistant' ? (
                                <div className="flex items-center space-x-2">
                                    <span className="animate-spin h-4 w-4 border-t-2 border-b-2 border-gray-500 rounded-full"></span>
                                    <span className="text-sm text-gray-600">Thinking...</span>
                                </div>
                            ) : (
                                <ReactMarkdown remarkPlugins={[remarkGfm]} components={
                                    { 
                                        code({ node, inline, className, children, ...props }) { 
                                            const match = /language-(\w+)/.exec(className || ''); 
                                            return !inline && match ? ( 
                                                <code className={`${className} block whitespace-pre-wrap bg-gray-800 text-white p-3 rounded-md font-mono text-sm overflow-x-auto`} {...props}>{String(children).replace(/\n$/, '')}</code>
                                            ) : ( 
                                                <code className={`${className} bg-gray-300 text-red-600 px-1 rounded font-mono text-sm`} {...props}>{children}</code>
                                            )
                                        }
                                    }
                                }>{displayContent}</ReactMarkdown>
                            )}

                            {/* Render Download Button using the derived showButton flag */}
                            {showButton && sessionIdForDownload && (
                                <button
                                    onClick={() => {
                                        console.log(`DOWNLOAD BUTTON CLICKED for session: ${sessionIdForDownload}`);

                                        const backendBaseUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8080';
                                        // Prepend the base URL to the API path
                                        const downloadUrl = `${backendBaseUrl}/api/download_project?session_id=${sessionIdForDownload}`;
                                        console.log(`Navigating to absolute download URL: ${downloadUrl}`);

                                        window.location.href = downloadUrl;
                                    }}
                                    className="mt-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium py-1.5 px-4 rounded-md shadow transition duration-150 ease-in-out self-start"
                                >
                                    Download Project (.zip)
                                </button>
                            )}
                        </div>

                        {/* --- Conditionally Render Iframe Below User Message --- */}
                        {message.role === 'user' && message.videoId && (
                            <div className="mt-3 mb-1 w-full max-w-xl lg:max-w-2xl flex justify-center"> {/* Centering container */}
                                {/* Aspect ratio container for 16:9 video */}
                                <div className="aspect-video w-full rounded-lg overflow-hidden shadow-md border border-gray-300">
                                    <iframe
                                        src={`https://www.youtube.com/embed/${message.videoId}`}
                                        title="YouTube video player"
                                        frameBorder="0"
                                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                        referrerPolicy="strict-origin-when-cross-origin"
                                        allowFullScreen
                                        className="w-full h-full"
                                    ></iframe>
                                </div>
                            </div>
                        )}
                        {/* --- End Iframe --- */}
                    </div>
                );
            })}
            <div ref={messagesEndRef} />
        </div>
    );
}