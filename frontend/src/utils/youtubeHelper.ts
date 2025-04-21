// src/utils/youtubeHelper.ts

export function extractYouTubeVideoId(url: string): string | null {
    if (!url) {
        return null;
    }
    let videoId: string | null = null;
    try {
        const parsedUrl = new URL(url);

        if (parsedUrl.hostname.includes('youtube.com') && parsedUrl.searchParams.has('v')) {
            videoId = parsedUrl.searchParams.get('v');
        }
        else if (parsedUrl.hostname.includes('youtu.be')) {
            videoId = parsedUrl.pathname.substring(1); 
        }
         else if (parsedUrl.hostname.includes('googleusercontent.com') && parsedUrl.pathname.includes('/youtube.com/')) {
             const parts = parsedUrl.pathname.split('/');
             const youtubeIndex = parts.indexOf('youtube.com');
             if (youtubeIndex !== -1 && parts.length > youtubeIndex + 1) {
                 videoId = parts[youtubeIndex + 1];
             }
         }

        if (videoId && /^[a-zA-Z0-9_-]{11}$/.test(videoId)) {
            return videoId;
        } else {
            const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/|googleusercontent\.com\/youtube\.com\/)([a-zA-Z0-9_-]{11})/;
            const match = url.match(regex);
            if (match && match[1]) {
                return match[1];
            }
            return null;
        }

    } catch (error) {
        console.error("Error parsing URL or extracting YouTube ID:", error);
        return null;
    }
}