// src/app/api/chat/route.ts
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
    console.log("API Route: Received request");

    try {
        const body = await request.json();
        const messageText: string | undefined = body.message;
        const sessionId: string | undefined = body.sessionId;

        if (!messageText) {
            console.error("API Route: Missing 'message' in request body.");
            return NextResponse.json({ error: "Missing 'message' text" }, { status: 400 });
        }
        if (!sessionId) {
             console.warn("API Route: Missing 'sessionId' in request body.");
        }

        console.log(`API Route: Processing message for session ${sessionId || 'N/A'}: "${messageText}"`);

        await new Promise(resolve => setTimeout(resolve, 500));

        let baseReply = `Backend processed message for session ${sessionId || 'N/A'}: "${messageText || ''}"`;
        const reply = `${baseReply}\nProject generation complete. [DOWNLOAD_PROJECT:${sessionId || 'unknown-session'}]`;

        console.log(`API Route: Sending reply string: "${reply}"`);
        return NextResponse.json({ reply: reply });

    } catch (error) {
        console.error("API Route: Error processing request:", error);
        let errorMessage = 'Internal Server Error';
        if (error instanceof Error) {
             errorMessage = error.message;
        } else if (typeof error === 'string') {
             errorMessage = error;
        }
        return NextResponse.json({ error: errorMessage }, { status: 500 });
    }
}