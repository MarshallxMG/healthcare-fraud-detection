import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { Activity, AlertTriangle, CheckCircle, ShieldAlert, DollarSign, Database, Search, Clock, UserCheck, Stethoscope, Globe, Bot, Send, Sparkles, FileText, MapPin, Zap } from 'lucide-react'
import { translations, t } from './translations'

// Configuration for API - uses environment variable in production, localhost in development
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [stats, setStats] = useState({ total_claims: 0, fraud_claims: 0, fraud_percentage: 0, inpatient_claims: 0, outpatient_claims: 0 });
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [lang, setLang] = useState('en'); // Language state: 'en' or 'hi'
  
  // Toggle language function
  const toggleLanguage = () => setLang(lang === 'en' ? 'hi' : 'en');
  
  // AI Assistant State
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m your Healthcare Fraud Detection AI Assistant. Ask me anything about fraud patterns, claim analysis, or investigation procedures.' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [aiInsights, setAiInsights] = useState(null);
  const [generatedReport, setGeneratedReport] = useState(null);
  
  // Agent AI State
  const [agentMessages, setAgentMessages] = useState([
    { role: 'assistant', content: '🤖 **Hello! I\'m the AI Fraud Investigator.**\n\nI can autonomously investigate fraud by querying the claims database, running the ML model, looking up disease pricing, and more.\n\nTry asking me:\n- *"What are the overall fraud statistics?"*\n- *"Investigate provider PRV51234"*\n- *"What\'s the expected price for diagnosis 4019 at a Government hospital?"*\n- *"Show me the top 10 highest-amount fraud claims"*', tools_used: [] }
  ]);
  const [agentInput, setAgentInput] = useState('');
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentSessionId, setAgentSessionId] = useState(null);
  
  // New Claim Form State
  const [newClaim, setNewClaim] = useState({
    provider_id: "PRV55001",
    provider_type: "Clinic",
    diagnosis_code: "4019",
    claim_type: "Outpatient",
    amount: 500.0,
    deductible: 50.0,
    num_diagnoses: 2,
    num_procedures: 1,
    length_of_stay: 0,
    patient_age: 65,
    chronic_conditions: 2
  });
  
  // Hospital Search State
  const [hospitalSearch, setHospitalSearch] = useState('');
  const [hospitalResults, setHospitalResults] = useState([]);
  const [showResults, setShowResults] = useState(false);

  // Debounced Hospital Search
  useEffect(() => {
    const searchHospitals = async () => {
      if (hospitalSearch.length < 2) {
        setHospitalResults([]);
        return;
      }
      
      try {
        const response = await axios.get(`${API_URL}/hospitals/search?query=${hospitalSearch}&type=${newClaim.provider_type}`);
        setHospitalResults(response.data.hospitals || []);
        setShowResults(true);
      } catch (error) {
        console.error("Error searching hospitals:", error);
      }
    };

    const timeoutId = setTimeout(() => {
      if (hospitalSearch) searchHospitals();
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [hospitalSearch, newClaim.provider_type]);

  const handleHospitalSelect = (hospital) => {
    setNewClaim({
      ...newClaim,
      provider_id: hospital.name, // Use name as ID for display
      provider_type: hospital.type
    });
    setHospitalSearch(hospital.name);
    setShowResults(false);
  };

  // Provider type benchmarks (Indian Healthcare Pricing)
  const benchmarks = {
    'Government': { avg: 1000, p95: 3000, color: 'text-green-400', icon: '🏛️', desc: 'AIIMS, Govt Hospital' },
    'Clinic': { avg: 5000, p95: 15000, color: 'text-yellow-400', icon: '🏥', desc: 'Private Clinic' },
    'Private': { avg: 10000, p95: 40000, color: 'text-purple-400', icon: '🏨', desc: 'Apollo, Fortis, Max' }
  };

  // Most Expensive Diseases (from data analysis)
  const expensiveDiseases = [
    { code: '51881', name: 'Acute Respiratory Failure', avgAmount: 17635 },
    { code: '4241', name: 'Aortic Valve Disorder', avgAmount: 16800 },
    { code: '51884', name: 'Acute & Chronic Resp Failure', avgAmount: 15246 },
    { code: '0389', name: 'Septicemia (Blood Poisoning)', avgAmount: 14872 },
    { code: '4414', name: 'Abdominal Aortic Aneurysm', avgAmount: 13765 },
    { code: '41071', name: 'Heart Attack (Subendo Infarct)', avgAmount: 13582 },
    { code: '03842', name: 'E. Coli Septicemia', avgAmount: 13473 },
    { code: '44024', name: 'Atherosclerosis w/ Gangrene', avgAmount: 12434 },
    { code: '99673', name: 'Kidney Dialysis Complications', avgAmount: 12152 },
    { code: '8208', name: 'Hip Fracture (Femur Neck)', avgAmount: 11561 }
  ];

  // Highest Fraud Rate Diseases (from data analysis)
  const dangerousDiseases = [
    { code: '44024', name: 'Atherosclerosis w/ Gangrene', fraudRate: 65.2 },
    { code: '03842', name: 'E. Coli Septicemia', fraudRate: 62.6 },
    { code: '8082', name: 'Fracture (Closed)', fraudRate: 58.4 },
    { code: '51881', name: 'Acute Respiratory Failure', fraudRate: 58.3 },
    { code: '51884', name: 'Acute & Chronic Resp Failure', fraudRate: 58.0 },
    { code: '82009', name: 'Hip Fracture', fraudRate: 57.5 },
    { code: '0389', name: 'Septicemia NOS', fraudRate: 57.0 },
    { code: '71536', name: 'Arthropathy', fraudRate: 57.0 },
    { code: '5070', name: 'Aspiration Pneumonia', fraudRate: 56.7 },
    { code: '486', name: 'Pneumonia', fraudRate: 56.6 }
  ];

  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const statsRes = await axios.get(`${API_URL}/stats`);
      const claimsRes = await axios.get(`${API_URL}/claims?limit=20`);
      setStats(statsRes.data);
      setClaims(claimsRes.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post(`${API_URL}/predict`, newClaim);
      setPrediction(res.data);
    } catch (error) {
      console.error("Prediction error:", error);
    }
  };

  // AI Chat Handler
  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;
    
    const userMessage = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setAiLoading(true);
    
    try {
      const res = await axios.post(`${API_URL}/ai/chat`, {
        message: chatInput,
        claim_context: prediction
      });
      
      if (res.data.success) {
        setChatMessages(prev => [...prev, { role: 'assistant', content: res.data.response }]);
      } else {
        setChatMessages(prev => [...prev, { role: 'assistant', content: '⚠️ AI service temporarily unavailable. Please try again later.' }]);
      }
    } catch (error) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Connection error. Please check if the backend is running.' }]);
    }
    setAiLoading(false);
  };

  // Fetch AI Insights
  const fetchAiInsights = async () => {
    setAiLoading(true);
    try {
      const res = await axios.get(`${API_URL}/ai/insights`);
      if (res.data.success) {
        setAiInsights(res.data.insights);
      }
    } catch (error) {
      console.error("Error fetching insights:", error);
    }
    setAiLoading(false);
  };

  // Generate AI Report
  const generateAiReport = async () => {
    if (!prediction) {
      alert('Please analyze a claim first!');
      return;
    }
    setAiLoading(true);
    try {
      const res = await axios.post(`${API_URL}/ai/report`, {
        claim_data: newClaim,
        prediction_result: prediction
      });
      if (res.data.success) {
        setGeneratedReport(res.data.report);
      }
    } catch (error) {
      console.error("Error generating report:", error);
    }
    setAiLoading(false);
  };

  // Agent Chat Handler
  const sendAgentMessage = async (messageOverride) => {
    const msg = messageOverride || agentInput;
    if (!msg.trim()) return;
    
    const userMessage = { role: 'user', content: msg, tools_used: [] };
    setAgentMessages(prev => [...prev, userMessage]);
    setAgentInput('');
    setAgentLoading(true);
    
    try {
      const res = await axios.post(`${API_URL}/agent/chat`, {
        message: msg,
        session_id: agentSessionId
      });
      
      if (res.data.success) {
        setAgentSessionId(res.data.session_id);
        setAgentMessages(prev => [...prev, {
          role: 'assistant',
          content: res.data.response,
          tools_used: res.data.tools_used || []
        }]);
      } else {
        setAgentMessages(prev => [...prev, {
          role: 'assistant',
          content: '⚠️ Agent error: ' + (res.data.response || 'Unknown error'),
          tools_used: []
        }]);
      }
    } catch (error) {
      setAgentMessages(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ Connection error. Please check if the backend is running on port 8000.',
        tools_used: []
      }]);
    }
    setAgentLoading(false);
  };

  // Tool name to emoji mapping
  const toolEmoji = (name) => {
    const map = {
      'query_claims_database': '🔍 Database Query',
      'run_fraud_prediction': '🧠 ML Prediction',
      'lookup_disease_price': '💰 Price Lookup',
      'get_provider_history': '📋 Provider History',
      'get_fraud_statistics': '📊 Fraud Stats',
      'search_hospital_info': '🏥 Hospital Search',
      'generate_investigation_report': '📝 Report Generated'
    };
    return map[name] || name;
  };

  const getRiskColor = (level) => {
    switch(level) {
      case 'Critical': return 'text-red-500 bg-red-500/20 border-red-500/30';
      case 'High': return 'text-orange-500 bg-orange-500/20 border-orange-500/30';
      case 'Medium': return 'text-yellow-500 bg-yellow-500/20 border-yellow-500/30';
      default: return 'text-green-500 bg-green-500/20 border-green-500/30';
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-cyan-500 selection:text-white">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-slate-800 border-r border-slate-700 p-6 flex flex-col">
        <div className="flex items-center gap-3 mb-10">
          <ShieldAlert className="w-8 h-8 text-cyan-400" />
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">
            {t('appName', lang)}
          </h1>
        </div>
        
        <nav className="space-y-2 flex-1">
          <button 
            onClick={() => setActiveTab('dashboard')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'dashboard' ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'}`}
          >
            <Activity className="w-5 h-5" />
            {t('liveMonitor', lang)}
          </button>
          <button 
            onClick={() => setActiveTab('analytics')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'analytics' ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'}`}
          >
            <Database className="w-5 h-5" />
            {t('analytics', lang)}
          </button>
          <button 
            onClick={() => setActiveTab('ai')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'ai' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'}`}
          >
            <Bot className="w-5 h-5" />
            <span className="flex items-center gap-2">
              AI Assistant
              <Sparkles className="w-3 h-3 text-yellow-400" />
            </span>
          </button>
          <button 
            onClick={() => setActiveTab('agent')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'agent' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'}`}
          >
            <Bot className="w-5 h-5 flex-shrink-0" />
            <span className="truncate">AI Investigator</span>
            <span className="text-[9px] px-1 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold flex-shrink-0 leading-none">NEW</span>
          </button>
        </nav>

        {/* Language Toggle */}
        <div className="mb-4">
          <button 
            onClick={toggleLanguage}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition-all"
          >
            <Globe className="w-4 h-4" />
            <span className="text-sm font-medium">
              {lang === 'en' ? 'हिंदी' : 'English'}
            </span>
          </button>
        </div>

        <div className="pt-6 border-t border-slate-700">
          <p className="text-xs text-slate-500 mb-2">{t('appTagline', lang)}</p>
          <div className="flex items-center gap-3 text-slate-400 text-sm">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            {stats.total_claims.toLocaleString()} {t('totalClaims', lang)}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="ml-64 p-8">
        {/* Header */}
        <header className="flex justify-between items-center mb-8">
          <div>
            <h2 className="text-2xl font-bold text-white mb-1">
              {activeTab === 'dashboard' ? 'Real-time Fraud Monitoring' : activeTab === 'agent' ? '🤖 AI Fraud Investigator' : activeTab === 'ai' ? 'AI Assistant' : 'System Analytics'}
            </h2>
            <p className="text-slate-400 text-sm">Powered by Medicare Claims Data • {stats.total_claims.toLocaleString()} records</p>
          </div>
          <div className="flex gap-4">
             <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 flex items-center gap-4">
               <div className="p-3 bg-blue-500/20 rounded-lg text-blue-400">
                 <Activity className="w-6 h-6" />
               </div>
               <div>
                 <p className="text-slate-400 text-xs uppercase tracking-wider">Total Claims</p>
                 <p className="text-xl font-bold text-white">{stats.total_claims.toLocaleString()}</p>
               </div>
             </div>
             
             <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 flex items-center gap-4">
               <div className="p-3 bg-red-500/20 rounded-lg text-red-400">
                 <AlertTriangle className="w-6 h-6" />
               </div>
               <div>
                 <p className="text-slate-400 text-xs uppercase tracking-wider">Fraud Rate</p>
                 <p className="text-xl font-bold text-white">{stats.fraud_percentage}%</p>
               </div>
             </div>
          </div>
        </header>

        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-3 gap-8">
            {/* Live Feed */}
            <div className="col-span-2 space-y-6">
              <div className="bg-slate-800 rounded-2xl border border-slate-700 overflow-hidden">
                <div className="p-6 border-b border-slate-700 flex justify-between items-center">
                  <h3 className="font-semibold text-white">Real Medicare Claims</h3>
                  <div className="flex gap-2">
                    <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full">
                      {stats.inpatient_claims?.toLocaleString() || 0} Inpatient
                    </span>
                    <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded-full">
                      {stats.outpatient_claims?.toLocaleString() || 0} Outpatient
                    </span>
                  </div>
                </div>
                <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                  <table className="w-full text-left">
                    <thead className="bg-slate-700/50 text-slate-400 text-xs uppercase sticky top-0 backdrop-blur-md">
                      <tr>
                        <th className="px-4 py-3 font-medium">Provider</th>
                        <th className="px-4 py-3 font-medium">Diagnosis</th>
                        <th className="px-4 py-3 font-medium">Type</th>
                        <th className="px-4 py-3 font-medium">Amount</th>
                        <th className="px-4 py-3 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700">
                      {claims.map((claim) => (
                        <tr key={claim.id} className="hover:bg-slate-700/30 transition-colors">
                          <td className="px-4 py-3">
                            <p className="text-slate-300 text-sm font-mono">{claim.provider_id}</p>
                            <p className="text-slate-500 text-xs">Age: {claim.patient_age || 'N/A'}</p>
                          </td>
                          <td className="px-4 py-3">
                            <p className="text-slate-300 text-sm truncate max-w-[180px]" title={claim.short_desc}>
                              {claim.short_desc || claim.diagnosis_code}
                            </p>
                            <p className="text-cyan-500 text-xs">{claim.diagnosis_code}</p>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`text-xs px-2 py-1 rounded-full ${claim.claim_type === 'Inpatient' ? 'bg-purple-500/20 text-purple-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                              {claim.claim_type}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-slate-300">${claim.amount?.toLocaleString()}</td>
                          <td className="px-4 py-3">
                            {claim.is_fraud ? (
                              <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-red-500/20 text-red-400 border border-red-500/20">
                                <AlertTriangle className="w-3 h-3" /> Fraud
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/20">
                                <CheckCircle className="w-3 h-3" /> Valid
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Manual Entry Simulator */}
            <div className="space-y-6">
              <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">
                <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                  <Stethoscope className="w-5 h-5 text-cyan-400" />
                  {t('analyzeClaim', lang)}
                </h3>
                
                <form onSubmit={handlePredict} className="space-y-3">
                  {/* Provider Type Selector */}
                  <div>
                    <label className="block text-xs text-slate-400 mb-2">{t('providerType', lang)}</label>
                    <div className="grid grid-cols-3 gap-2">
                      {Object.entries(benchmarks).map(([type, info]) => (
                        <button
                          key={type}
                          type="button"
                          onClick={() => setNewClaim({...newClaim, provider_type: type})}
                          className={`p-2 rounded-lg border text-xs font-medium transition-all ${
                            newClaim.provider_type === type 
                              ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400' 
                              : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-slate-600'
                          }`}
                        >
                          <span className="text-lg">{info.icon}</span>
                          <p className="mt-1">{type}</p>
                          <p className="text-[10px] text-slate-500">avg ₹{info.avg.toLocaleString()}</p>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Hospital Name</label>
                      <div className="relative">
                        <div className="relative">
                          <input 
                            type="text" 
                            value={hospitalSearch}
                            onChange={(e) => {
                              setHospitalSearch(e.target.value);
                              setShowResults(true);
                            }}
                            placeholder="Type hospital name..."
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
                          />
                          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                        </div>
                        
                        {/* Autocomplete Dropdown */}
                        {showResults && hospitalResults.length > 0 && (
                          <div className="absolute z-50 w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl max-h-60 overflow-y-auto no-scrollbar">
                            {hospitalResults.map((hospital, idx) => (
                              <button
                                key={idx}
                                type="button"
                                onClick={() => handleHospitalSelect(hospital)}
                                className="w-full text-left px-3 py-2 hover:bg-slate-700 transition-colors flex items-start gap-2 border-b border-slate-700/50 last:border-0"
                              >
                                <MapPin className="w-3 h-3 text-cyan-500 mt-1 shrink-0" />
                                <div>
                                  <p className="text-sm text-slate-200 font-medium">{hospital.name}</p>
                                  <p className="text-[10px] text-slate-400">
                                    {hospital.district}, {hospital.state} • {hospital.type}
                                  </p>
                                </div>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">{t('diagnosisCode', lang)}</label>
                      <input type="text" value={newClaim.diagnosis_code}
                        onChange={(e) => setNewClaim({...newClaim, diagnosis_code: e.target.value})}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">{t('amount', lang)} (₹)</label>
                      <input type="number" value={newClaim.amount}
                        onChange={(e) => setNewClaim({...newClaim, amount: parseFloat(e.target.value)})}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">{t('stayDays', lang)}</label>
                      <input type="number" min="0" value={newClaim.length_of_stay}
                        onChange={(e) => {
                          const stay = parseInt(e.target.value) || 0;
                          setNewClaim({
                            ...newClaim, 
                            length_of_stay: stay,
                            claim_type: stay >= 1 ? "Inpatient" : "Outpatient"
                          });
                        }}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
                      />
                      <p className={`text-xs mt-1 ${newClaim.length_of_stay >= 1 ? 'text-purple-400' : 'text-emerald-400'}`}>
                        → {newClaim.length_of_stay >= 1 ? `🏥 ${t('inpatient', lang)}` : `🚶 ${t('outpatient', lang)}`}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">{t('diagnoses', lang)}</label>
                      <input type="number" value={newClaim.num_diagnoses}
                        onChange={(e) => setNewClaim({...newClaim, num_diagnoses: parseInt(e.target.value)})}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">{t('patientAge', lang)}</label>
                      <input type="number" min="0" max="120" value={newClaim.patient_age}
                        onChange={(e) => {
                          const age = parseInt(e.target.value) || 0;
                          setNewClaim({...newClaim, patient_age: age});
                        }}
                        className={`w-full bg-slate-900 border rounded-lg px-3 py-2 text-sm text-white focus:outline-none ${
                          newClaim.patient_age > 120 || newClaim.patient_age < 0 
                            ? 'border-red-500 focus:border-red-500' 
                            : 'border-slate-700 focus:border-cyan-500'
                        }`}
                      />
                      {(newClaim.patient_age > 120 || newClaim.patient_age < 0) && (
                        <p className="text-xs text-red-400 mt-1">⚠️ {t('invalidAge', lang)}</p>
                      )}
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs text-slate-400 mb-1">{t('chronicConditions', lang)} (0-11)</label>
                    <input type="range" min="0" max="11" value={newClaim.chronic_conditions}
                      onChange={(e) => setNewClaim({...newClaim, chronic_conditions: parseInt(e.target.value)})}
                      className="w-full"
                    />
                    <p className="text-center text-sm text-slate-400">{newClaim.chronic_conditions} {t('conditions', lang)}</p>
                  </div>
                  
                  <button type="submit"
                    disabled={newClaim.patient_age > 120 || newClaim.patient_age < 0}
                    className={`w-full font-semibold py-2.5 rounded-lg shadow-lg transition-all active:scale-[0.98] ${
                      newClaim.patient_age > 120 || newClaim.patient_age < 0
                        ? 'bg-slate-600 text-slate-400 cursor-not-allowed shadow-none'
                        : 'bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white shadow-cyan-500/25'
                    }`}
                  >
                    {newClaim.patient_age > 120 || newClaim.patient_age < 0 ? `⚠️ ${t('invalidAge', lang)}` : t('analyzeRisk', lang)}
                  </button>
                </form>

                {prediction && (
                  <div className={`mt-4 p-4 rounded-xl border ${prediction.is_fraud ? 'bg-red-500/10 border-red-500/20' : 'bg-green-500/10 border-green-500/20'}`}>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        {prediction.is_fraud ? (
                          <AlertTriangle className="w-5 h-5 text-red-500" />
                        ) : (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        )}
                        <span className={`font-bold ${prediction.is_fraud ? 'text-red-400' : 'text-green-400'}`}>
                          {prediction.is_fraud ? 'Fraud Risk Detected' : 'Low Risk'}
                        </span>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full border ${getRiskColor(prediction.risk_level)}`}>
                        {prediction.risk_level}
                      </span>
                    </div>
                    
                    <p className="text-sm text-cyan-400 mb-1">{prediction.short_desc}</p>
                    <p className="text-xs text-slate-500 mb-3">{prediction.category_desc}</p>
                    
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-sm text-slate-400">
                        Risk Score: <span className="font-bold">{(prediction.probability * 100).toFixed(1)}%</span>
                      </p>
                      {prediction.detection_method && (
                        <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded-full border border-purple-500/20">
                          {prediction.detection_method.includes('Rule') ? '📋 Rules' : '🤖 ML'}
                        </span>
                      )}
                    </div>
                    
                    {prediction.detection_method && (
                      <p className="text-xs text-slate-500 mb-2">{prediction.detection_method}</p>
                    )}
                    
                    {/* Provider Benchmark Comparison */}
                    {prediction.benchmark_info && (
                      <div className="mt-2 p-2 bg-slate-900 rounded-lg">
                        <p className="text-xs text-slate-400 mb-1">📊 Benchmark Comparison ({prediction.provider_type})</p>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-slate-500">Expected avg:</span>
                            <span className="text-slate-300 ml-1">₹{prediction.benchmark_info.expected_average?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">95th percentile:</span>
                            <span className="text-slate-300 ml-1">₹{prediction.benchmark_info.p95_threshold?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Your amount:</span>
                            <span className={`ml-1 font-bold ${prediction.is_fraud ? 'text-red-400' : 'text-green-400'}`}>
                              ₹{prediction.benchmark_info.your_amount?.toLocaleString()}
                            </span>
                          </div>
                          <div>
                            <span className={`px-1 py-0.5 rounded text-[10px] ${
                              prediction.benchmark_info.comparison === 'Above 95th percentile' 
                                ? 'bg-red-500/20 text-red-400'
                                : prediction.benchmark_info.comparison === 'Above average'
                                ? 'bg-yellow-500/20 text-yellow-400'
                                : 'bg-green-500/20 text-green-400'
                            }`}>
                              {prediction.benchmark_info.comparison}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* GST Breakdown */}
                    {prediction.gst_info && (
                      <div className="mt-2 p-2 bg-gradient-to-r from-orange-500/10 to-yellow-500/10 rounded-lg border border-orange-500/20">
                        <p className="text-xs text-orange-400 mb-1 font-medium">🧾 GST Breakdown (18%)</p>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div>
                            <span className="text-slate-500">Base:</span>
                            <span className="text-slate-300 ml-1">₹{prediction.gst_info.base_amount?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">GST:</span>
                            <span className="text-orange-400 ml-1">+₹{prediction.gst_info.gst_amount?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Total:</span>
                            <span className="text-green-400 ml-1 font-bold">₹{prediction.gst_info.total_with_gst?.toLocaleString()}</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Disease-Specific Pricing Zone */}
                    {prediction.price_zone_info && prediction.expected_price_info && (
                      <div className={`mt-2 p-3 rounded-lg border ${
                        prediction.price_zone_info.zone === 'Normal' 
                          ? 'bg-green-500/10 border-green-500/30' 
                          : prediction.price_zone_info.zone === 'Elevated'
                          ? 'bg-yellow-500/10 border-yellow-500/30'
                          : 'bg-red-500/10 border-red-500/30'
                      }`}>
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-xs font-medium flex items-center gap-1">
                            <span>{prediction.price_zone_info.emoji}</span>
                            <span className={
                              prediction.price_zone_info.zone === 'Normal' ? 'text-green-400' 
                              : prediction.price_zone_info.zone === 'Elevated' ? 'text-yellow-400' 
                              : 'text-red-400'
                            }>
                              {prediction.price_zone_info.zone} Price Zone
                            </span>
                          </p>
                          <span className={`text-xs px-2 py-0.5 rounded-full ${
                            prediction.price_zone_info.zone === 'Normal' 
                              ? 'bg-green-500/20 text-green-400' 
                              : prediction.price_zone_info.zone === 'Elevated'
                              ? 'bg-yellow-500/20 text-yellow-400'
                              : 'bg-red-500/20 text-red-400'
                          }`}>
                            {prediction.price_zone_info.ratio}x expected
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2 text-xs mb-2">
                          <div>
                            <span className="text-slate-500">Base (disease):</span>
                            <span className="text-slate-300 ml-1">₹{prediction.expected_price_info.base_price?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Expected ({newClaim.provider_type}):</span>
                            <span className="text-cyan-400 ml-1">₹{prediction.expected_price_info.expected_without_gst?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Normal max:</span>
                            <span className="text-green-400 ml-1">₹{prediction.expected_price_info.max_normal?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Elevated max:</span>
                            <span className="text-yellow-400 ml-1">₹{prediction.expected_price_info.max_elevated?.toLocaleString()}</span>
                          </div>
                        </div>
                        
                        <p className="text-xs text-slate-400 italic">
                          {prediction.price_zone_info.explanation}
                        </p>
                      </div>
                    )}
                    
                    {prediction.risk_factors && prediction.risk_factors.length > 0 && prediction.risk_factors[0] !== "No specific risk factors detected" && (
                      <div className="mt-3 pt-3 border-t border-slate-700">
                        <p className="text-xs text-slate-400 mb-2">Risk Factors:</p>
                        <ul className="space-y-1">
                          {prediction.risk_factors.map((factor, i) => (
                            <li key={i} className="text-xs text-orange-400 flex items-start gap-1">
                              <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" /> 
                              <span>{factor}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="grid grid-cols-2 gap-8">
            {/* Fraud Distribution */}
            <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">
              <h3 className="font-semibold text-white mb-6">Fraud Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Legitimate', value: stats.total_claims - stats.fraud_claims },
                      { name: 'Fraudulent', value: stats.fraud_claims }
                    ]}
                    cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={5} dataKey="value"
                  >
                    <Cell fill="#22c55e" />
                    <Cell fill="#ef4444" />
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#ffffff' }} 
                    itemStyle={{ color: '#ffffff' }}
                    labelStyle={{ color: '#ffffff' }}
                  />
                  <Legend wrapperStyle={{ color: '#ffffff' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Key Metrics */}
            <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">
              <h3 className="font-semibold text-white mb-6">Dataset Statistics</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-4 bg-slate-900 rounded-xl">
                  <span className="text-slate-400">Total Claims</span>
                  <span className="text-2xl font-bold text-white">{stats.total_claims.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-slate-900 rounded-xl">
                  <span className="text-slate-400">Fraudulent Providers</span>
                  <span className="text-2xl font-bold text-red-400">{stats.fraud_claims.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-slate-900 rounded-xl">
                  <span className="text-slate-400">Inpatient Claims</span>
                  <span className="text-2xl font-bold text-purple-400">{stats.inpatient_claims?.toLocaleString() || 0}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-slate-900 rounded-xl">
                  <span className="text-slate-400">Outpatient Claims</span>
                  <span className="text-2xl font-bold text-emerald-400">{stats.outpatient_claims?.toLocaleString() || 0}</span>
                </div>
              </div>
            </div>

            {/* Claim Types */}
            <div className="col-span-2 bg-slate-800 rounded-2xl border border-slate-700 p-6">
              <h3 className="font-semibold text-white mb-6">Claims by Type</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { type: 'Inpatient', legitimate: Math.round((stats.inpatient_claims || 0) * (1 - stats.fraud_percentage/100)), fraud: Math.round((stats.inpatient_claims || 0) * stats.fraud_percentage/100) },
                  { type: 'Outpatient', legitimate: Math.round((stats.outpatient_claims || 0) * (1 - stats.fraud_percentage/100)), fraud: Math.round((stats.outpatient_claims || 0) * stats.fraud_percentage/100) }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="type" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
                  <Legend />
                  <Bar dataKey="legitimate" name="Legitimate" fill="#22c55e" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="fraud" name="Fraudulent" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Most Expensive Diseases */}
            <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">
              <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-yellow-400" />
                💰 Most Expensive Diseases
              </h3>
              <div className="overflow-hidden rounded-lg border border-slate-700">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900">
                    <tr>
                      <th className="text-left py-2 px-3 text-slate-400 font-medium">#</th>
                      <th className="text-left py-2 px-3 text-slate-400 font-medium">Code</th>
                      <th className="text-left py-2 px-3 text-slate-400 font-medium">Disease</th>
                      <th className="text-right py-2 px-3 text-slate-400 font-medium">Avg Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {expensiveDiseases.map((d, i) => (
                      <tr key={d.code} className="border-t border-slate-700 hover:bg-slate-700/50">
                        <td className="py-2 px-3 text-slate-500">{i + 1}</td>
                        <td className="py-2 px-3 font-mono text-cyan-400">{d.code}</td>
                        <td className="py-2 px-3 text-white">{d.name}</td>
                        <td className="py-2 px-3 text-right font-bold text-yellow-400">₹{d.avgAmount.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Highest Fraud Rate Diseases */}
            <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">
              <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                🚨 Highest Fraud Rate Diseases
              </h3>
              <div className="overflow-hidden rounded-lg border border-slate-700">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900">
                    <tr>
                      <th className="text-left py-2 px-3 text-slate-400 font-medium">#</th>
                      <th className="text-left py-2 px-3 text-slate-400 font-medium">Code</th>
                      <th className="text-left py-2 px-3 text-slate-400 font-medium">Disease</th>
                      <th className="text-right py-2 px-3 text-slate-400 font-medium">Fraud Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dangerousDiseases.map((d, i) => (
                      <tr key={d.code} className="border-t border-slate-700 hover:bg-slate-700/50">
                        <td className="py-2 px-3 text-slate-500">{i + 1}</td>
                        <td className="py-2 px-3 font-mono text-cyan-400">{d.code}</td>
                        <td className="py-2 px-3 text-white">{d.name}</td>
                        <td className="py-2 px-3 text-right">
                          <span className={`font-bold ${d.fraudRate > 60 ? 'text-red-400' : 'text-orange-400'}`}>
                            {d.fraudRate}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* AI Assistant Tab - Temporarily Disabled */}
        {activeTab === 'ai' && (
          <div className="text-center py-12">
            <Bot className="w-16 h-16 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-slate-400 mb-2">Legacy AI Assistant</h3>
            <p className="text-slate-500 mb-6">This has been replaced by the new AI Investigator.</p>
            <button onClick={() => setActiveTab('agent')} className="px-6 py-3 bg-emerald-500 hover:bg-emerald-400 text-white rounded-lg transition-all font-medium">
              Go to AI Investigator →
            </button>
          </div>
        )}

        {/* 🤖 AGENTIC AI INVESTIGATOR TAB */}
        {activeTab === 'agent' && (
          <div className="grid grid-cols-4 gap-6" style={{height: 'calc(100vh - 180px)'}}>
            {/* Chat Panel - Takes 3 columns */}
            <div className="col-span-3 bg-slate-800 rounded-2xl border border-slate-700 flex flex-col overflow-hidden">
              {/* Chat Header */}
              <div className="p-4 border-b border-slate-700 flex items-center justify-between bg-gradient-to-r from-emerald-500/10 to-cyan-500/10">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-emerald-500/20 rounded-lg">
                    <Zap className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white text-sm">AI Fraud Investigator</h3>
                    <p className="text-xs text-slate-400">Autonomous agent with 7 investigation tools</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {agentSessionId && (
                    <span className="text-[10px] px-2 py-1 bg-slate-700 text-slate-400 rounded-full font-mono">
                      Session: {agentSessionId.slice(0, 8)}...
                    </span>
                  )}
                  <button
                    onClick={() => {
                      setAgentMessages([agentMessages[0]]);
                      setAgentSessionId(null);
                    }}
                    className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-all"
                  >
                    New Session
                  </button>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                {agentMessages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] ${msg.role === 'user' ? 'order-2' : ''}`}>
                      {/* Tool badges */}
                      {msg.tools_used && msg.tools_used.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mb-2">
                          {msg.tools_used.map((tool, tIdx) => (
                            <span key={tIdx} className="text-[10px] px-2 py-1 bg-emerald-500/15 text-emerald-400 rounded-full border border-emerald-500/20 font-medium">
                              {toolEmoji(tool.tool)}
                            </span>
                          ))}
                        </div>
                      )}
                      {/* Message bubble */}
                      <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                        msg.role === 'user'
                          ? 'bg-cyan-500/20 text-cyan-50 border border-cyan-500/20 rounded-br-md'
                          : 'bg-slate-700/60 text-slate-200 border border-slate-600/40 rounded-bl-md'
                      }`}>
                        {/* Render markdown-like bold and lists */}
                        {msg.content.split('\n').map((line, lIdx) => {
                          // Bold
                          let rendered = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                          // Italic  
                          rendered = rendered.replace(/\*(.*?)\*/g, '<em>$1</em>');
                          // Inline code
                          rendered = rendered.replace(/`(.*?)`/g, '<code class="bg-slate-800 px-1 py-0.5 rounded text-cyan-400 text-xs">$1</code>');
                          
                          if (line.trim().startsWith('- ') || line.trim().startsWith('• ')) {
                            return <div key={lIdx} className="pl-3 py-0.5" dangerouslySetInnerHTML={{__html: '• ' + rendered.replace(/^[\s]*[-•]\s*/, '')}} />;
                          }
                          if (line.trim().match(/^\d+\.\s/)) {
                            return <div key={lIdx} className="pl-3 py-0.5" dangerouslySetInnerHTML={{__html: rendered}} />;
                          }
                          if (line.trim().startsWith('#')) {
                            const level = line.match(/^#+/)[0].length;
                            const text = rendered.replace(/^#+\s*/, '');
                            const sizes = { 1: 'text-lg font-bold', 2: 'text-base font-semibold', 3: 'text-sm font-semibold' };
                            return <div key={lIdx} className={`${sizes[level] || 'font-semibold'} mt-2 mb-1 text-white`} dangerouslySetInnerHTML={{__html: text}} />;
                          }
                          if (line.trim() === '') return <div key={lIdx} className="h-2" />;
                          return <div key={lIdx} className="py-0.5" dangerouslySetInnerHTML={{__html: rendered}} />;
                        })}
                      </div>
                    </div>
                  </div>
                ))}
                {agentLoading && (
                  <div className="flex justify-start">
                    <div className="bg-slate-700/60 border border-slate-600/40 rounded-2xl rounded-bl-md px-4 py-3">
                      <div className="flex items-center gap-3">
                        <div className="flex gap-1">
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                        </div>
                        <span className="text-xs text-slate-400">Agent is investigating...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Input */}
              <div className="p-4 border-t border-slate-700 bg-slate-800/80">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={agentInput}
                    onChange={(e) => setAgentInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !agentLoading && sendAgentMessage()}
                    placeholder="Ask the AI Investigator anything..."
                    disabled={agentLoading}
                    className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/30 transition-all disabled:opacity-50"
                  />
                  <button
                    onClick={() => sendAgentMessage()}
                    disabled={agentLoading || !agentInput.trim()}
                    className="px-5 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-white rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-emerald-500/20 active:scale-95"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Actions Sidebar */}
            <div className="col-span-1 space-y-4">
              {/* Quick Actions */}
              <div className="bg-slate-800 rounded-2xl border border-slate-700 p-4">
                <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  Quick Actions
                </h4>
                <div className="space-y-2">
                  {[
                    { label: '📊 Fraud Statistics', prompt: 'What are the overall fraud statistics?' },
                    { label: '🔍 Top Fraud Claims', prompt: 'Show me the top 10 highest-amount fraudulent claims' },
                    { label: '💰 Price Check', prompt: 'What is the expected price for diagnosis code 4019 (Hypertension) at a Government hospital vs Private hospital?' },
                    { label: '🏥 Hospital Search', prompt: 'Search for AIIMS hospitals' },
                    { label: '📋 Provider Probe', prompt: 'Investigate provider PRV51234 - show their full claim history and check for suspicious patterns' },
                    { label: '🧠 Analyze Claim', prompt: 'Run fraud prediction on a claim: provider PRV001, Government hospital, diagnosis 4019, amount ₹25000, patient age 65' },
                  ].map((action, idx) => (
                    <button
                      key={idx}
                      onClick={() => sendAgentMessage(action.prompt)}
                      disabled={agentLoading}
                      className="w-full text-left px-3 py-2.5 bg-slate-900 hover:bg-slate-700 border border-slate-700 hover:border-slate-600 rounded-lg text-xs text-slate-300 hover:text-white transition-all disabled:opacity-50"
                    >
                      {action.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Agent Capabilities */}
              <div className="bg-slate-800 rounded-2xl border border-slate-700 p-4">
                <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-purple-400" />
                  Agent Tools
                </h4>
                <div className="space-y-1.5">
                  {[
                    { emoji: '🔍', name: 'Database Query' },
                    { emoji: '🧠', name: 'ML Prediction' },
                    { emoji: '💰', name: 'Price Lookup' },
                    { emoji: '📋', name: 'Provider History' },
                    { emoji: '📊', name: 'Fraud Statistics' },
                    { emoji: '🏥', name: 'Hospital Search' },
                    { emoji: '📝', name: 'Report Generator' },
                  ].map((tool, idx) => (
                    <div key={idx} className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs text-slate-400">
                      <span>{tool.emoji}</span>
                      <span>{tool.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
