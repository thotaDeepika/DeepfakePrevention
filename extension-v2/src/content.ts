function injectSafeUploadButtons() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach((input) => {
        // Cast to HTMLInputElement to access properties
        const htmlInput = input as HTMLInputElement;
        
        if (htmlInput.hasAttribute('data-safeupload-injected')) return;
        htmlInput.setAttribute('data-safeupload-injected', 'true');

        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.display = 'inline-block';

        const btn = document.createElement('button');
        btn.innerHTML = '🛡️ Cloak with SafeUpload';
        btn.style.cssText = `
            position: absolute;
            bottom: -35px;
            left: 0;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            z-index: 999999;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            white-space: nowrap;
        `;

        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();

            // 1. Create a hidden file input to let the user pick a photo
            const hiddenInput = document.createElement('input');
            hiddenInput.type = 'file';
            hiddenInput.accept = 'image/jpeg, image/png, image/webp';
            
            hiddenInput.onchange = async (e) => {
                const target = e.target as HTMLInputElement;
                if (!target.files || target.files.length === 0) return;
                
                const file = target.files[0];
                btn.innerHTML = '⏳ Cloaking... (Takes 30-60s)';
                btn.style.opacity = '0.7';
                btn.disabled = true;

                try {
                    // 2. Send the file to the local SafeUpload backend
                    const formData = new FormData();
                    formData.append('image', file);

                    const response = await fetch('http://127.0.0.1:7860/protect', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) throw new Error('Backend failed to cloak image.');

                    const data = await response.json();
                    
                    // 3. Convert the returned base64 back into a File object
                    const byteCharacters = atob(data.protected_b64);
                    const byteNumbers = new Array(byteCharacters.length);
                    for (let i = 0; i < byteCharacters.length; i++) {
                        byteNumbers[i] = byteCharacters.charCodeAt(i);
                    }
                    const byteArray = new Uint8Array(byteNumbers);
                    const blob = new Blob([byteArray], { type: 'image/jpeg' });
                    
                    const cloakedFile = new File([blob], `safeupload_${file.name}`, {
                        type: 'image/jpeg',
                        lastModified: new Date().getTime()
                    });

                    // 4. Inject the cloaked file into the website's original input!
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(cloakedFile);
                    htmlInput.files = dataTransfer.files;

                    // 5. Trigger change events so the website's React/Angular code detects it
                    htmlInput.dispatchEvent(new Event('change', { bubbles: true }));
                    htmlInput.dispatchEvent(new Event('input', { bubbles: true }));

                    btn.innerHTML = '✅ Cloaked & Injected!';
                    btn.style.background = '#10b981';
                    
                    setTimeout(() => {
                        btn.innerHTML = '🛡️ Cloak with SafeUpload';
                        btn.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
                        btn.style.opacity = '1';
                        btn.disabled = false;
                    }, 3000);

                } catch (err: any) {
                    alert('SafeUpload Error: ' + err.message);
                    btn.innerHTML = '❌ Failed';
                    btn.style.background = '#ef4444';
                    setTimeout(() => {
                        btn.innerHTML = '🛡️ Cloak with SafeUpload';
                        btn.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
                        btn.style.opacity = '1';
                        btn.disabled = false;
                    }, 3000);
                }
            };
            
            hiddenInput.click();
        });

        const parent = htmlInput.parentNode;
        if (parent) {
            parent.insertBefore(wrapper, htmlInput);
            wrapper.appendChild(htmlInput);
            wrapper.appendChild(btn);
        }
    });
}

// Run initially and set up observer for dynamically added inputs
injectSafeUploadButtons();
injectTwitterButton();

const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
        if (mutation.addedNodes.length) {
            injectSafeUploadButtons();
            injectTwitterButton();
        }
    }
});

observer.observe(document.body, { childList: true, subtree: true });

// Specific injection for Twitter / X
function injectTwitterButton() {
    if (!window.location.hostname.includes('twitter.com') && !window.location.hostname.includes('x.com')) return;
    
    // Twitter's media button
    const mediaBtns = document.querySelectorAll('[aria-label="Add photos or video"]');
    mediaBtns.forEach(mediaBtn => {
        if (mediaBtn.hasAttribute('data-safeupload-injected')) return;
        mediaBtn.setAttribute('data-safeupload-injected', 'true');

        const btn = document.createElement('button');
        btn.innerHTML = '🛡️ SafeUpload';
        btn.style.cssText = `
            background: #8b5cf6 !important;
            color: #ffffff !important;
            border: 2px solid #a855f7 !important;
            border-radius: 9999px !important;
            padding: 0 16px !important;
            font-size: 14px !important;
            font-weight: 800 !important;
            cursor: pointer !important;
            margin-left: 12px !important;
            height: 34px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            outline: none !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
        `;

        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();

            const hiddenInput = document.createElement('input');
            hiddenInput.type = 'file';
            hiddenInput.accept = 'image/jpeg, image/png, image/webp';
            
            hiddenInput.onchange = async (e) => {
                const target = e.target as HTMLInputElement;
                if (!target.files || target.files.length === 0) return;
                
                const file = target.files[0];
                const originalText = btn.innerHTML;
                btn.innerHTML = '⏳ Cloaking...';
                btn.style.opacity = '0.7';
                btn.disabled = true;

                try {
                    // Content script doesn't close on click away, so fetch is safe!
                    const formData = new FormData();
                    formData.append('image', file);

                    const response = await fetch('http://127.0.0.1:7860/protect', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) throw new Error('Backend failed');

                    const data = await response.json();
                    
                    const byteCharacters = atob(data.protected_b64);
                    const byteNumbers = new Array(byteCharacters.length);
                    for (let i = 0; i < byteCharacters.length; i++) {
                        byteNumbers[i] = byteCharacters.charCodeAt(i);
                    }
                    const byteArray = new Uint8Array(byteNumbers);
                    const blob = new Blob([byteArray], { type: 'image/jpeg' });
                    
                    const cloakedFile = new File([blob], `safeupload_${file.name}`, {
                        type: 'image/jpeg',
                        lastModified: new Date().getTime()
                    });

                    // Inject into Twitter's hidden input
                    const twitterInput = document.querySelector('input[data-testid="fileInput"]') as HTMLInputElement;
                    if (twitterInput) {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(cloakedFile);
                        twitterInput.files = dataTransfer.files;
                        twitterInput.dispatchEvent(new Event('change', { bubbles: true }));
                    } else {
                        // Fallback download if input is missing
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `safeupload_${file.name}`;
                        a.click();
                        URL.revokeObjectURL(url);
                    }

                    btn.innerHTML = '✅ Injected!';
                    btn.style.background = '#10b981';
                    btn.style.opacity = '1';
                    
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                        btn.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
                        btn.disabled = false;
                    }, 3000);

                } catch (err: any) {
                    btn.innerHTML = '❌ Failed';
                    btn.style.background = '#ef4444';
                    btn.style.opacity = '1';
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                        btn.style.background = 'linear-gradient(135deg, #3b82f6, #8b5cf6)';
                        btn.disabled = false;
                    }, 3000);
                }
            };
            hiddenInput.click();
        });

        // Insert next to the media button
        if (mediaBtn.parentNode) {
            mediaBtn.parentNode.insertBefore(btn, mediaBtn.nextSibling);
        }
    });
}

