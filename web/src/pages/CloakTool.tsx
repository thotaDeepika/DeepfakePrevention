import React, { useState, useRef } from 'react';
import { UploadCloud, Shield, Download, RefreshCw, Activity, Lock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleCloak = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      // In production, we'd use the Celery/SSE endpoint. 
      // For now, we hit the proxy to the existing synchronous endpoint.
      const res = await fetch('/protect', {
        method: 'POST',
        body: formData,
      });
      
      const data = await res.json();
      
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Failed to process image');
      }
      
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatChartData = (metrics: any) => {
    if (!metrics) return [];
    return Object.keys(metrics).map((model) => ({
      model: model.toUpperCase(),
      Reduction: metrics[model].reduction_pct,
      Original: metrics[model].original_sim * 100,
      Protected: metrics[model].protected_sim * 100,
    }));
  };

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-6xl mx-auto flex flex-col gap-8">
      {/* Header */}
      <header className="text-center pt-8 pb-4">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="inline-block px-4 py-1.5 rounded-full bg-primaryDim border border-primary/30 text-primary2 text-xs font-bold uppercase tracking-widest mb-4">
          Defensive Privacy Tool
        </motion.div>
        <motion.h1 initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="text-4xl md:text-5xl font-black mb-4 bg-gradient-to-br from-primary2 to-accent bg-clip-text text-transparent">
          SafeUpload
        </motion.h1>
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="text-slate-400 max-w-2xl mx-auto text-sm md:text-base">
          Adversarial identity cloaking across FaceNet, ArcFace, and CLIP. Reduces AI identity consistency while preserving human-recognizable appearance.
        </motion.p>
      </header>

      {/* Main Content Area */}
      <main className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left Column: Upload & Actions */}
        <div className="flex flex-col gap-6">
          <div className="bg-surface border border-border rounded-2xl p-6 shadow-xl">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white text-sm">1</span>
              Upload Photo
            </h2>
            
            <div 
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => !isProcessing && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${
                isProcessing ? 'opacity-50 cursor-not-allowed border-border2' : 'border-border2 hover:border-primary hover:bg-primaryDim/50'
              }`}
            >
              <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" disabled={isProcessing} />
              
              <AnimatePresence mode="wait">
                {preview ? (
                  <motion.div key="preview" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="relative max-h-64 mx-auto w-fit">
                    <img src={preview} alt="Preview" className="max-h-64 rounded-lg shadow-lg border border-border" />
                    {!isProcessing && (
                      <div className="absolute inset-0 bg-black/50 opacity-0 hover:opacity-100 flex items-center justify-center rounded-lg transition-opacity">
                        <p className="text-white font-semibold">Change Image</p>
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-4 py-8">
                    <div className="p-4 bg-surface3 rounded-full text-primary2">
                      <UploadCloud size={40} />
                    </div>
                    <div>
                      <p className="font-semibold text-slate-200">Tap to upload or drag and drop</p>
                      <p className="text-xs text-slate-400 mt-1">JPG, PNG, WEBP (Max 10MB)</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {error && (
              <div className="mt-4 p-4 rounded-lg bg-danger/10 border border-danger/30 text-danger text-sm">
                {error}
              </div>
            )}

            <button 
              onClick={handleCloak} 
              disabled={!file || isProcessing}
              className={`w-full mt-6 py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-lg ${
                !file || isProcessing 
                  ? 'bg-surface3 text-slate-400 cursor-not-allowed shadow-none' 
                  : 'bg-gradient-to-r from-primary to-accent text-white hover:scale-[1.02] hover:shadow-primary/20'
              }`}
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="animate-spin" size={20} />
                  Cloaking Identity...
                </>
              ) : (
                <>
                  <Shield size={20} />
                  Cloak Identity
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="flex flex-col gap-6">
          <AnimatePresence mode="wait">
            {isProcessing && !result && (
              <motion.div key="loading" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="bg-surface border border-border rounded-2xl p-8 flex flex-col items-center justify-center min-h-[400px] text-center shadow-xl">
                <div className="w-16 h-16 relative mb-6">
                  <div className="absolute inset-0 border-4 border-primary/30 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
                <h3 className="text-xl font-bold mb-2">Generating Adversarial Cloak</h3>
                <p className="text-slate-400 text-sm max-w-xs">
                  Applying multi-crop alignment and identity diversion. This usually takes 30-90 seconds depending on GPU availability.
                </p>
              </motion.div>
            )}

            {result && (
              <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-6">
                
                {/* Result Card */}
                <div className="bg-surface border border-border rounded-2xl p-6 shadow-xl overflow-hidden relative">
                  <div className="absolute top-0 right-0 p-4 opacity-10">
                    <Lock size={120} />
                  </div>
                  <h2 className="text-lg font-bold mb-6 flex items-center gap-2 relative z-10">
                    <span className="w-8 h-8 rounded-full bg-gradient-to-br from-success to-emerald-500 flex items-center justify-center text-white text-sm">2</span>
                    Protected Image Ready
                  </h2>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 relative z-10">
                    <div className="flex flex-col gap-2">
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Original</span>
                      <img src={`data:image/jpeg;base64,${result.original_b64}`} alt="Original" className="rounded-xl border border-border" />
                    </div>
                    <div className="flex flex-col gap-2">
                      <span className="text-xs font-semibold text-success uppercase tracking-wider">Protected</span>
                      <img src={`data:image/jpeg;base64,${result.protected_b64}`} alt="Protected" className="rounded-xl border border-success/50 shadow-[0_0_15px_rgba(34,197,94,0.15)]" />
                    </div>
                  </div>

                  <a 
                    href={`/download/${result.job_id}`}
                    download={`safeupload_${result.job_id}.jpg`}
                    className="w-full py-3 rounded-xl bg-success/10 border border-success/30 text-success font-bold flex items-center justify-center gap-2 hover:bg-success hover:text-white transition-colors z-10 relative"
                  >
                    <Download size={20} />
                    Download Protected Image
                  </a>
                </div>

                {/* Metrics Card */}
                <div className="bg-surface border border-border rounded-2xl p-6 shadow-xl">
                   <h2 className="text-lg font-bold mb-6 flex items-center gap-2">
                    <Activity size={24} className="text-primary2" />
                    Identity Reduction Metrics
                  </h2>
                  
                  <div className="h-64 w-full mb-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="50%" outerRadius="70%" data={formatChartData(result.metrics)}>
                        <PolarGrid stroke="#2a3655" />
                        <PolarAngleAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                        <Radar name="Original Similarity" dataKey="Original" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} />
                        <Radar name="Protected Similarity" dataKey="Protected" stroke="#22c55e" fill="#22c55e" fillOpacity={0.5} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#2a3655', borderRadius: '8px' }} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.keys(result.metrics).map((model) => (
                      <div key={model} className="bg-surface2 rounded-xl p-3 border border-border">
                        <div className="text-[10px] uppercase text-primary2 font-bold mb-1">{model}</div>
                        <div className="text-xl font-black text-white">
                          -{result.metrics[model].reduction_pct.toFixed(0)}%
                        </div>
                        <div className="text-xs text-slate-400 mt-1">Consistency</div>
                      </div>
                    ))}
                  </div>
                </div>

              </motion.div>
            )}
            
            {!isProcessing && !result && (
              <div className="bg-surface/50 border border-border/50 rounded-2xl p-8 flex flex-col items-center justify-center min-h-[400px] text-center border-dashed text-slate-500">
                <Shield size={48} className="mb-4 opacity-50" />
                <p>Upload a photo to see protection metrics and results here.</p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

export default App;
