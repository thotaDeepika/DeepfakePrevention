const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const resultImg = document.getElementById('result-img');
const uploadText = document.getElementById('upload-text');
const cloakBtn = document.getElementById('cloak-btn');
const downloadBtn = document.getElementById('download-btn');
const statusText = document.getElementById('status');
const loader = document.getElementById('loader');

let selectedFile = null;
let resultDataUrl = null;

// Check if a process is already running in the background when popup opens
chrome.storage.local.get(['isProcessing'], (result) => {
    if (result.isProcessing) {
        uploadArea.style.display = 'none';
        cloakBtn.style.display = 'none';
        downloadBtn.style.display = 'none';
        loader.style.display = 'block';
        statusText.innerText = "🛡️ Shielding in progress...\n\nYour terminal is currently crunching the AI math. Please wait for the download box to appear. (Do not upload another image yet!)";
        statusText.style.color = "#8b5cf6";
    }
});

uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            uploadText.style.display = 'none';
            cloakBtn.style.display = 'block';
            resultImg.style.display = 'none';
            downloadBtn.style.display = 'none';
            statusText.innerText = '';
        };
        reader.readAsDataURL(selectedFile);
    }
});

cloakBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    cloakBtn.style.display = 'none';
    loader.style.display = 'block';
    
    // UI update to let the user know they can close it safely
    statusText.innerText = "🚀 Shielding started!\n\nYou can safely close this popup now. The image will automatically download when it finishes.";
    statusText.style.color = "#10b981";
    uploadArea.style.display = 'none'; // Hide upload box so they don't click it again

    const reader = new FileReader();
    reader.onload = (e) => {
        // Send the image to the background script so it survives popup closure
        chrome.runtime.sendMessage({
            action: 'start_cloaking',
            dataUrl: e.target.result
        });
    };
    reader.readAsDataURL(selectedFile);
});

downloadBtn.addEventListener('click', () => {
    if (!resultDataUrl) return;
    chrome.downloads.download({
        url: resultDataUrl,
        filename: 'safeupload_protected.jpg'
    });
});
