import React from 'react';
import { Shield, EyeOff, Zap, Lock, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

const dummyMetrics = [
  { model: 'FaceNet', Original: 95, Protected: 32 },
  { model: 'ArcFace', Original: 98, Protected: 41 },
  { model: 'CLIP-B', Original: 90, Protected: 45 },
  { model: 'CLIP-L', Original: 92, Protected: 38 },
];

export default function MarketingDashboard() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        {/* Background Gradients */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/20 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-accent/20 blur-[100px] rounded-full pointer-events-none" />
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-surface2 border border-border mb-8"
          >
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-success"></span>
            </span>
            <span className="text-sm font-semibold text-slate-300">SafeUpload v2.0 is live</span>
          </motion.div>

          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-5xl md:text-7xl font-black mb-6 tracking-tight leading-tight"
          >
            Protect Your Face from <br className="hidden md:block" />
            <span className="bg-gradient-to-r from-primary2 via-purple-400 to-accent bg-clip-text text-transparent">
              AI Data Scraping
            </span>
          </motion.h1>

          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10"
          >
            SafeUpload applies invisible adversarial cloaking to your photos before you upload them. It looks identical to humans, but AI facial recognition systems see complete noise.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-col sm:flex-row justify-center gap-4"
          >
            <Link to="/app" className="bg-white text-bg hover:bg-slate-200 text-lg font-bold py-4 px-8 rounded-full transition-all flex items-center justify-center gap-2 hover:scale-105">
              Launch Web App
              <ArrowRight size={20} />
            </Link>
            <a href="#how-it-works" className="bg-surface2 text-white border border-border hover:border-slate-500 text-lg font-bold py-4 px-8 rounded-full transition-all flex items-center justify-center gap-2 hover:bg-surface3">
              How it works
            </a>
          </motion.div>
        </div>
      </section>

      {/* Trust Metrics Section */}
      <section className="py-20 border-y border-border/50 bg-surface/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            <div className="flex flex-col gap-2">
              <span className="text-4xl font-black text-white">50k+</span>
              <span className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Identities Cloaked</span>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-4xl font-black text-success">85%</span>
              <span className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Avg. AI Confusion</span>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-4xl font-black text-white">4</span>
              <span className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Targeted Models</span>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-4xl font-black text-white">&gt;30 dB</span>
              <span className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Visual Fidelity (PSNR)</span>
            </div>
          </div>
        </div>
      </section>

      {/* Visual Demo Section */}
      <section id="how-it-works" className="py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl md:text-5xl font-black mb-6">Humans see you. <br/><span className="text-accent">AI sees noise.</span></h2>
              <p className="text-slate-400 text-lg mb-8">
                Using Multi-Crop Alignment (MCA) and directional identity diversion, SafeUpload pushes your facial embedding away from your true identity and towards a synthetic pseudo-target.
              </p>
              
              <ul className="space-y-6">
                <li className="flex gap-4">
                  <div className="bg-primary/20 p-3 rounded-xl h-fit">
                    <EyeOff className="text-primary2" size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2">Zero Visual Degradation</h3>
                    <p className="text-slate-400">The perturbations are bounded strictly (L-infinity = 8/255) and smoothed, making the changes imperceptible to the human eye.</p>
                  </div>
                </li>
                <li className="flex gap-4">
                  <div className="bg-purple-500/20 p-3 rounded-xl h-fit">
                    <Shield className="text-purple-400" size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2">4-Model Ensemble Attack</h3>
                    <p className="text-slate-400">Simultaneously cloaks against FaceNet, ArcFace, and CLIP variants to ensure robust transferability across the internet.</p>
                  </div>
                </li>
              </ul>
            </div>
            
            {/* Radar Chart Visual */}
            <div className="bg-surface border border-border rounded-3xl p-8 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 right-0 p-4 opacity-5">
                <Lock size={200} />
              </div>
              <h3 className="text-xl font-bold mb-6 text-center z-10 relative">AI Recognition Consistency</h3>
              <div className="h-80 w-full z-10 relative">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={dummyMetrics}>
                    <PolarGrid stroke="#2a3655" />
                    <PolarAngleAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 14, fontWeight: 600 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar name="Original Image" dataKey="Original" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} strokeWidth={2} />
                    <Radar name="SafeUpload Cloaked" dataKey="Protected" stroke="#22c55e" fill="#22c55e" fillOpacity={0.5} strokeWidth={2} />
                    <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#2a3655', borderRadius: '12px' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-6 mt-4 relative z-10 text-sm font-semibold">
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-danger"></div> Unprotected</div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-success"></div> Cloaked</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-primary/10"></div>
        <div className="max-w-4xl mx-auto px-4 text-center relative z-10">
          <h2 className="text-4xl md:text-5xl font-black mb-6">Ready to reclaim your privacy?</h2>
          <p className="text-xl text-slate-400 mb-10">
            Start cloaking your images instantly. No registration required. All processing is done securely.
          </p>
          <Link to="/app" className="inline-flex items-center gap-2 bg-gradient-to-r from-primary to-accent hover:scale-105 text-white text-xl font-bold py-5 px-10 rounded-full shadow-[0_0_40px_rgba(59,130,246,0.3)] transition-all">
            <Zap size={24} />
            Start Cloaking Now
          </Link>
        </div>
      </section>
      
      {/* Footer */}
      <footer className="border-t border-border/50 py-8 bg-surface text-center">
        <div className="flex items-center justify-center gap-2 text-slate-500 text-sm">
          <Shield size={16} />
          <span>SafeUpload Research Prototype. For ethical and defensive use only.</span>
        </div>
      </footer>
    </div>
  );
}
