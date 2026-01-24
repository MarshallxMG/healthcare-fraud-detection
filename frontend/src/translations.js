// Language translations for Healthcare Fraud Detection System
// Supports: English (en) and Hindi (hi)

export const translations = {
  en: {
    // App Title
    appName: "FraudGuard",
    appTagline: "Healthcare Fraud Detection",
    
    // Navigation
    liveMonitor: "Live Monitor",
    analytics: "Analytics",
    analyzeClaim: "Analyze Claim",
    dataset: "Dataset",
    
    // Dashboard Stats
    totalClaims: "Total Claims",
    fraudDetected: "Fraud Detected",
    inpatient: "Inpatient",
    outpatient: "Outpatient",
    
    // Analyze Claim Form
    providerType: "Provider Type",
    government: "Government",
    clinic: "Clinic",
    private: "Private",
    providerId: "Provider ID",
    diagnosisCode: "Diagnosis Code",
    amount: "Amount",
    stayDays: "Stay (days)",
    diagnoses: "Diagnoses",
    patientAge: "Patient Age",
    chronicConditions: "Chronic Conditions",
    conditions: "conditions",
    analyzeRisk: "Analyze Risk",
    
    // Provider Descriptions
    govtDesc: "AIIMS, Govt Hospital",
    clinicDesc: "Private Clinic",
    privateDesc: "Apollo, Fortis, Max",
    
    // Risk Levels
    lowRisk: "Low Risk",
    mediumRisk: "Medium Risk",
    highRisk: "High Risk",
    criticalRisk: "Critical Risk",
    riskScore: "Risk Score",
    
    // Prediction Results
    fraudDetectedTitle: "Fraud Detected",
    legitimateClaim: "Legitimate Claim",
    riskFactors: "Risk Factors",
    detectionMethod: "Detection Method",
    
    // GST & Pricing
    gstBreakdown: "GST Breakdown (18%)",
    baseAmount: "Base Amount",
    gstAmount: "GST Amount",
    totalWithGst: "Total with GST",
    
    // Price Zone
    diseaseSpecificPricing: "Disease-Specific Pricing",
    expectedPrice: "Expected Price",
    priceZone: "Price Zone",
    normal: "Normal",
    elevated: "Elevated",
    suspicious: "Suspicious",
    
    // Tables
    recentTransactions: "Recent Transactions",
    claimId: "Claim ID",
    disease: "Disease",
    type: "Type",
    status: "Status",
    timestamp: "Timestamp",
    
    // Analytics
    fraudDistribution: "Fraud Distribution",
    fraudCases: "Fraud Cases",
    legitimateCases: "Legitimate Cases",
    expensiveDiseases: "Most Expensive Diseases",
    highFraudRateDiseases: "High Fraud Rate Diseases",
    avgAmount: "Avg Amount",
    fraudRate: "Fraud Rate",
    
    // Messages
    invalidAge: "Invalid age (must be 0-120)",
    analysisComplete: "Analysis complete",
    loading: "Loading...",
    
    // Language Toggle
    language: "Language",
    english: "English",
    hindi: "हिंदी"
  },
  
  hi: {
    // App Title
    appName: "फ्रॉडगार्ड",
    appTagline: "स्वास्थ्य धोखाधड़ी पहचान",
    
    // Navigation
    liveMonitor: "लाइव मॉनिटर",
    analytics: "विश्लेषण",
    analyzeClaim: "दावा विश्लेषण",
    dataset: "डेटासेट",
    
    // Dashboard Stats
    totalClaims: "कुल दावे",
    fraudDetected: "धोखाधड़ी पाई गई",
    inpatient: "भर्ती मरीज",
    outpatient: "बाहरी मरीज",
    
    // Analyze Claim Form
    providerType: "प्रदाता प्रकार",
    government: "सरकारी",
    clinic: "क्लिनिक",
    private: "निजी",
    providerId: "प्रदाता आईडी",
    diagnosisCode: "निदान कोड",
    amount: "राशि",
    stayDays: "ठहराव (दिन)",
    diagnoses: "निदान",
    patientAge: "मरीज की आयु",
    chronicConditions: "पुरानी बीमारियां",
    conditions: "बीमारियां",
    analyzeRisk: "जोखिम विश्लेषण करें",
    
    // Provider Descriptions
    govtDesc: "एम्स, सरकारी अस्पताल",
    clinicDesc: "निजी क्लिनिक",
    privateDesc: "अपोलो, फोर्टिस, मैक्स",
    
    // Risk Levels
    lowRisk: "कम जोखिम",
    mediumRisk: "मध्यम जोखिम",
    highRisk: "उच्च जोखिम",
    criticalRisk: "गंभीर जोखिम",
    riskScore: "जोखिम स्कोर",
    
    // Prediction Results
    fraudDetectedTitle: "धोखाधड़ी पाई गई",
    legitimateClaim: "वैध दावा",
    riskFactors: "जोखिम कारक",
    detectionMethod: "पहचान विधि",
    
    // GST & Pricing
    gstBreakdown: "जीएसटी विवरण (18%)",
    baseAmount: "मूल राशि",
    gstAmount: "जीएसटी राशि",
    totalWithGst: "जीएसटी सहित कुल",
    
    // Price Zone
    diseaseSpecificPricing: "रोग-विशिष्ट मूल्य निर्धारण",
    expectedPrice: "अपेक्षित मूल्य",
    priceZone: "मूल्य क्षेत्र",
    normal: "सामान्य",
    elevated: "बढ़ा हुआ",
    suspicious: "संदिग्ध",
    
    // Tables
    recentTransactions: "हाल के लेनदेन",
    claimId: "दावा आईडी",
    disease: "बीमारी",
    type: "प्रकार",
    status: "स्थिति",
    timestamp: "समय",
    
    // Analytics
    fraudDistribution: "धोखाधड़ी वितरण",
    fraudCases: "धोखाधड़ी मामले",
    legitimateCases: "वैध मामले",
    expensiveDiseases: "सबसे महंगी बीमारियां",
    highFraudRateDiseases: "उच्च धोखाधड़ी दर वाली बीमारियां",
    avgAmount: "औसत राशि",
    fraudRate: "धोखाधड़ी दर",
    
    // Messages
    invalidAge: "अमान्य आयु (0-120 होनी चाहिए)",
    analysisComplete: "विश्लेषण पूर्ण",
    loading: "लोड हो रहा है...",
    
    // Language Toggle
    language: "भाषा",
    english: "English",
    hindi: "हिंदी"
  }
};

// Helper function to get translation
export const t = (key, lang = 'en') => {
  return translations[lang]?.[key] || translations['en'][key] || key;
};

export default translations;
