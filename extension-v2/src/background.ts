chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'cloak_image') {
    handleCloakImage(message.dataBase64, message.filename)
      .then((result) => sendResponse({ status: 'success' }))
      .catch((error) => sendResponse({ status: 'error', message: error.toString() }));
    
    return true; // Indicates async response
  }
});

async function handleCloakImage(base64Data: string, filename: string) {
  // 1. Convert base64 back to Blob
  const byteString = atob(base64Data.split(',')[1]);
  const mimeString = base64Data.split(',')[0].split(':')[1].split(';')[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
  }
  const blob = new Blob([ab], { type: mimeString });

  // 2. Send to backend
  const formData = new FormData();
  formData.append('image', blob, filename);

  const response = await fetch('http://127.0.0.1:7860/protect', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Server returned an error. Is the backend running?');
  }

  const data = await response.json();

  // 3. Convert response base64 back to Blob
  const outByteCharacters = atob(data.protected_b64);
  const outByteNumbers = new Array(outByteCharacters.length);
  for (let i = 0; i < outByteCharacters.length; i++) {
    outByteNumbers[i] = outByteCharacters.charCodeAt(i);
  }
  const outByteArray = new Uint8Array(outByteNumbers);
  const outBlob = new Blob([outByteArray], { type: 'image/jpeg' });

  // 4. Download it via Chrome Downloads API
  const reader = new FileReader();
  reader.onloadend = () => {
    const dataUrl = reader.result as string;
    chrome.downloads.download({
      url: dataUrl,
      filename: `safeupload_${filename}`,
      saveAs: false
    });
  };
  reader.readAsDataURL(outBlob);
}
