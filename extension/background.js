chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'start_cloaking') {
        chrome.storage.local.set({ isProcessing: true });
        processImage(request.dataUrl);
        sendResponse({status: "started"});
    }
    return true;
});

async function processImage(dataUrl) {
    try {
        // Convert the base64 data URL back into a Blob
        const res = await fetch(dataUrl);
        const blob = await res.blob();
        
        const formData = new FormData();
        formData.append('image', blob, 'image.jpg');

        console.log("SafeUpload: Sending image to local server...");

        const response = await fetch('http://127.0.0.1:7860/protect', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.protected_b64) {
            const resultDataUrl = "data:image/jpeg;base64," + data.protected_b64;
            // Automatically download the protected image
            chrome.downloads.download({
                url: resultDataUrl,
                filename: 'safeupload_protected.jpg',
                saveAs: true // Prompts the user where to save it
            });
            console.log("SafeUpload: Download triggered successfully.");
        } else {
            console.error("SafeUpload API Error:", data.error || "Failed to process image.");
        }
    } catch (error) {
        console.error("SafeUpload Fetch Error: Make sure the Python server is running.", error);
    } finally {
        // Mark as finished so the popup unlocks
        chrome.storage.local.set({ isProcessing: false });
    }
}
