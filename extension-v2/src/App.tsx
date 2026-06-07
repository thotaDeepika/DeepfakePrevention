import React, { useState, useRef } from 'react';
import { Shield, Upload, Activity, CheckCircle, AlertCircle } from 'lucide-react';
import './index.css';

export default function App() {
  const [status, setStatus] = useState('Idle');
  const [file, setFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setStatus('Idle');
      setErrorMsg('');
    }
  };

  const handleCloak = async () => {
    if (!file) {
      setErrorMsg('Please select an image first.');
      return;
    }
    
    setStatus('Processing...');
    setErrorMsg('');

    try {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64Data = reader.result as string;
        
        // Send to background script
        chrome.runtime.sendMessage({
          action: 'cloak_image',
          filename: file.name,
          dataBase64: base64Data
        }, (response) => {
          if (chrome.runtime.lastError) {
             setErrorMsg('Extension error: ' + chrome.runtime.lastError.message);
             setStatus('Error');
             return;
          }
          if (response && response.status === 'error') {
            setErrorMsg(response.message || 'Failed to cloak image.');
            setStatus('Error');
          } else {
            // We just assume success here, the download happens in background
            setStatus('Cloaked Successfully!');
          }
        });
        
        // We set status to processing, and tell the user they can close the popup
        setStatus('Processing in Background...');
      };
      reader.readAsDataURL(file);

    } catch (err: any) {
      setStatus('Error');
      setErrorMsg(err.message || 'Failed to initialize cloak.');
    }
  };

  return (
    <div className="w-80 p-6 bg-slate-900 text-slate-200 font-sans shadow-2xl relative">
      <div className="flex items-center justify-center gap-2 mb-6">
        <Shield className="text-purple-500" size={28} />
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          SafeUpload
        </h1>
      </div>

      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept="image/jpeg, image/png, image/webp" 
        className="hidden" 
      />

      <div 
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors cursor-pointer mb-4 group ${
          file ? 'border-success bg-success/10' : 'border-slate-700 hover:border-purple-500 hover:bg-slate-800'
        }`}
      >
        {file ? (
          <>
            <CheckCircle className="mx-auto text-success mb-2" size={32} />
            <p className="font-semibold text-sm text-success truncate px-2">{file.name}</p>
          </>
        ) : (
          <>
            <Upload className="mx-auto text-slate-500 group-hover:text-purple-400 transition-colors mb-2" size={32} />
            <p className="font-semibold text-sm">Select Image to Cloak</p>
          </>
        )}
      </div>

      {errorMsg && (
        <div className="flex items-center gap-2 text-danger text-xs mb-4 bg-danger/10 p-2 rounded-lg">
          <AlertCircle size={14} />
          <span>{errorMsg}</span>
        </div>
      )}

      <button 
        onClick={handleCloak}
        disabled={status === 'Processing...'}
        className={`w-full font-bold py-3 rounded-xl shadow-lg transition-transform flex items-center justify-center gap-2 ${
          status === 'Processing...' 
            ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
            : status === 'Cloaked Successfully!'
            ? 'bg-success text-white'
            : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white hover:scale-105'
        }`}
      >
        <Activity size={18} className={status === 'Processing...' ? 'animate-spin' : ''} />
        {status === 'Idle' ? 'Cloak Identity' : status}
      </button>

      <p className="text-center text-xs text-slate-500 mt-4">
        Downloads automatically when finished.
      </p>
    </div>
  );
}
