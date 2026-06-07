import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import MarketingDashboard from './pages/MarketingDashboard';
import CloakTool from './pages/CloakTool';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-bg text-slate-200 font-sans flex flex-col selection:bg-primary selection:text-white">
        <NavBar />
        <div className="flex-1">
          <Routes>
            <Route path="/" element={<MarketingDashboard />} />
            <Route path="/app" element={<CloakTool />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
