import { CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import { useTheme } from './ThemeProvider';

export default function ResultsPanel({ data, formData }) {
  const { isDark } = useTheme();
  const { 
    status, 
    risk_category, 
    approval_probability, 
    risk_score, 
    key_factors 
  } = data;

  // Determine colors based on status
  let statusColor = "text-red-500";
  let badgeBg = "bg-red-500/10 text-red-500 border border-red-500/20";
  let Icon = XCircle;
  let shadowGlowClass = isDark ? "shadow-verdict-rejected" : "shadow-card-light";

  if (status === "APPROVED") {
    statusColor = "text-green-500";
    badgeBg = "bg-green-500/10 text-green-500 border border-green-500/20";
    Icon = CheckCircle2;
    shadowGlowClass = isDark ? "shadow-verdict-approved" : "shadow-card-light";
  } else if (status === "CONDITIONAL") {
    statusColor = "text-yellow-500";
    badgeBg = "bg-yellow-500/10 text-yellow-500 border border-yellow-500/20";
    Icon = AlertCircle;
  }

  // Circular gauge calculations
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (risk_score / 100) * circumference;
  
  // Decide color for risk ring
  let riskRingColor = "#FF5A4E"; // coral/red for high risk
  if (risk_score < 25) riskRingColor = "#10b981"; // green
  else if (risk_score < 50) riskRingColor = "#f59e0b"; // yellow

  return (
    <div className={`mt-4 bg-card rounded-3xl p-8 relative z-10 scale-[1.02] transform transition-all duration-700 ease-out animate-in fade-in slide-in-from-bottom-8 ${shadowGlowClass}`}>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-inputBorder pb-6 mb-8 gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-textMuted font-bold mb-2">Decision</div>
          <div className={`flex items-center gap-2 px-5 py-2 rounded-full font-bold tracking-wide ${badgeBg}`}>
            <Icon size={20} />
            {status}
          </div>
        </div>
        <div className="md:text-right">
          <div className="text-xs uppercase tracking-wider text-textMuted font-bold mb-1">Risk Category</div>
          <div className="text-2xl font-bold font-display text-textMain">{risk_category}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-10">
        
        {/* Risk Score Circular Gauge */}
        <div className="flex items-center gap-8">
          <div className="relative w-32 h-32 flex-shrink-0">
            <svg className="w-full h-full transform -rotate-90 drop-shadow-md" viewBox="0 0 80 80">
              <circle cx="40" cy="40" r={radius} fill="none" stroke={isDark ? "#1F1F1F" : "#E5E0D8"} strokeWidth="6" />
              <circle 
                cx="40" cy="40" r={radius} fill="none" 
                stroke={riskRingColor} 
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className={`text-3xl font-extrabold font-display ${isDark ? 'text-gradient' : 'text-textMain'}`}>{risk_score}</span>
            </div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-textMuted font-bold mb-2">Risk Score</div>
            <p className="text-sm text-textMuted font-medium leading-relaxed max-w-[200px]">Score from 0-100 indicating credit risk. Lower is better.</p>
          </div>
        </div>

        {/* Approval Probability Bar */}
        <div className="flex flex-col justify-center">
          <div className="flex justify-between items-end mb-3">
            <div className="text-xs uppercase tracking-wider text-textMuted font-bold">Approval Probability</div>
            <div className={`text-3xl font-extrabold font-display ${isDark ? 'text-gradient' : 'text-textMain'}`}>{approval_probability}%</div>
          </div>
          <div className="h-4 w-full bg-inputBg rounded-full overflow-hidden border border-inputBorder">
            <div 
              className="h-full bg-gradient-to-r from-secondaryAccent to-accent rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${approval_probability}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Key Factors */}
      <div>
        <div className="text-xs uppercase tracking-wider text-textMuted font-bold mb-4">Key factors influencing this decision</div>
        <div className="space-y-4">
          {key_factors.map((factor, idx) => (
            <div key={idx} className="flex items-center gap-4">
              <div className={`w-2 h-8 rounded-full shadow-sm ${factor.type === 'positive' ? 'bg-green-500' : 'bg-[#FF5A4E]'}`}></div>
              <span className="text-textMain font-medium">{factor.factor}</span>
            </div>
          ))}
        </div>
        
        <div className="mt-8 p-6 bg-inputBg rounded-2xl border border-inputBorder relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-accent to-secondaryAccent"></div>
          <p className="text-base text-textMain font-medium leading-relaxed italic ml-2">
            "Approval likelihood is {approval_probability >= 75 ? 'high' : approval_probability >= 50 ? 'moderate' : 'low'} due to {key_factors[0]?.factor.toLowerCase()}."
          </p>
        </div>
      </div>
    </div>
  );
}
