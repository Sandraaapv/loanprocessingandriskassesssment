import { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import AssessmentForm from './components/AssessmentForm';
import ModelPerformance from './components/ModelPerformance';
import Documentation from './components/Documentation';
import { useTheme } from './components/ThemeProvider';

function LiveTicker() {
  const mockData = [
    "Applicant #1042 → Approved · 91%",
    "Applicant #8921 → Conditional · 64%",
    "Applicant #3391 → Rejected · 12%",
    "Applicant #5502 → Approved · 88%"
  ];
  const [index, setIndex] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % mockData.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="hidden sm:flex items-center gap-3 bg-card backdrop-blur-xl border border-border shadow-sm px-5 py-2 rounded-full text-xs font-medium">
      <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></div>
      <div className="relative overflow-hidden h-4 w-52">
        <AnimatePresence mode="wait">
          <motion.div
            key={index}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -20, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 text-textMuted"
          >
            {mockData[index]}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('assessment');
  const { isDark, toggleTheme } = useTheme();

  return (
    <>
      <div className="aurora-bg">
        <div className="aurora-blob aurora-blob-1"></div>
        <div className="aurora-blob aurora-blob-2"></div>
      </div>
      
      <div className="min-h-screen text-textMain font-sans p-4 sm:p-8">
        <div className="max-w-6xl mx-auto relative z-10">
          <motion.header 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-10 flex flex-col sm:flex-row sm:items-start justify-between gap-4"
          >
            <div>
              <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight mb-2 font-display">
                <span className="text-gradient">AI-powered</span> risk assessment
              </h1>
              <p className="text-textMuted text-sm sm:text-base font-medium">Automated credit evaluation model</p>
            </div>
            
            <div className="flex items-center gap-4">
              <LiveTicker />
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={toggleTheme}
                className="relative overflow-hidden p-2.5 rounded-full bg-card border border-border text-textMuted hover:text-accent transition-colors focus:outline-none focus:ring-2 focus:ring-accent/50"
                aria-label="Toggle Dark Mode"
              >
                <AnimatePresence mode="wait">
                  <motion.div
                    key={isDark ? 'dark' : 'light'}
                    initial={{ rotate: -90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: 90, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    {isDark ? <Sun size={18} /> : <Moon size={18} />}
                  </motion.div>
                </AnimatePresence>
              </motion.button>
            </div>
          </motion.header>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="border-b border-border mb-10 flex gap-8 overflow-x-auto no-scrollbar"
          >
            {['assessment', 'performance', 'documentation'].map((tab) => (
              <button 
                key={tab}
                className={`pb-4 text-sm font-bold whitespace-nowrap transition-all duration-200 relative ${activeTab === tab ? 'text-textMain' : 'text-textMuted hover:text-textMain'}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'assessment' ? 'Applicant Assessment' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                {activeTab === tab && (
                  <motion.div 
                    layoutId="activeTabIndicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-accent to-secondaryAccent rounded-t-full"
                  />
                )}
              </button>
            ))}
          </motion.div>

          <main className="relative">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
              >
                {activeTab === 'assessment' && <AssessmentForm />}
                {activeTab === 'performance' && <ModelPerformance />}
                {activeTab === 'documentation' && <Documentation />}
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>
    </>
  );
}

export default App;
