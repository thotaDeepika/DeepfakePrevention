import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Shield, ExternalLink } from 'lucide-react';
import { motion } from 'framer-motion';

const NavBar = () => {
  const location = useLocation();
  const isApp = location.pathname === '/app';

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border/50 bg-bg/80 backdrop-blur-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center gap-2 group">
            <motion.div whileHover={{ rotate: 15 }} className="bg-primary/20 p-2 rounded-lg">
              <Shield className="text-primary2" size={24} />
            </motion.div>
            <span className="font-black text-xl bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent group-hover:to-white transition-colors">
              SafeUpload
            </span>
          </Link>
          
          <div className="flex items-center gap-6">
            <Link 
              to="/" 
              className={`text-sm font-semibold transition-colors ${!isApp ? 'text-white' : 'text-slate-400 hover:text-white'}`}
            >
              Overview
            </Link>
            
            <Link 
              to="/app" 
              className="bg-gradient-to-r from-primary to-accent hover:from-primary2 hover:to-purple-500 text-white text-sm font-bold py-2 px-5 rounded-full shadow-lg hover:shadow-primary/20 transition-all flex items-center gap-2"
            >
              Launch App
              <ExternalLink size={16} />
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
