import React, { useState } from 'react';
import { Shield, Upload, Activity } from 'lucide-react';
import './index.css';

export default function App() {
  const [status, setStatus] = useState('Idle');

  const handleCloak = async () => {
    setStatus('Processing...');
    // In a real extension, this would send a message to background.js
    // which handles the Celery polling API.
    setTimeout(() => setStatus('Cloaked Successfully!'), 2000);
  };

  return (
    <div className="w-80 p-6 bg-slate-900 text-slate-200 font-sans shadow-2xl">
      <div className="flex items-center justify-center gap-2 mb-6">
        <Shield className="text-purple-500" size={28} />
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          SafeUpload Shield
        </h1>
      </div>

      <div className="border-2 border-dashed border-slate-700 rounded-xl p-6 text-center hover:border-purple-500 hover:bg-slate-800 transition-colors cursor-pointer mb-6 group">
        <Upload className="mx-auto text-slate-500 group-hover:text-purple-400 transition-colors mb-2" size={32} />
        <p className="font-semibold text-sm">Select Image to Cloak</p>
      </div>

      <button 
        onClick={handleCloak}
        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold py-3 rounded-xl shadow-lg transition-transform hover:scale-105 flex items-center justify-center gap-2"
      >
        <Activity size={18} />
        {status === 'Idle' ? 'Cloak Identity' : status}
      </button>

      <p className="text-center text-xs text-slate-500 mt-4">
        Auto-detection on social media is active.
      </p>
    </div>
  );
}
