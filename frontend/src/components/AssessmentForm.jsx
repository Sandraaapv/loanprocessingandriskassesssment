import { useState, useMemo } from 'react';
import { User, Wallet, ShieldCheck, Check } from 'lucide-react';
import ResultsPanel from './ResultsPanel';

const PillToggle = ({ options, value, onChange, name }) => (
  <div className="flex bg-inputBg p-1 rounded-xl border border-inputBorder w-full sm:w-auto overflow-x-auto no-scrollbar">
    {options.map((opt) => (
      <button
        key={opt}
        type="button"
        onClick={() => onChange({ target: { name, value: opt } })}
        className={`flex-1 min-w-[80px] text-sm px-4 py-2 rounded-lg font-medium transition-all duration-200 whitespace-nowrap ${
          value === opt 
            ? 'bg-accent text-white shadow-sm' 
            : 'text-textMuted hover:text-textMain hover:bg-black/5 dark:hover:bg-white/5'
        }`}
      >
        {opt}
      </button>
    ))}
  </div>
);

export default function AssessmentForm() {
  const [formData, setFormData] = useState({
    Gender: '',
    Married: '',
    Dependents: '',
    Education: '',
    Self_Employed: '',
    ApplicantIncome: '',
    CoapplicantIncome: '',
    LoanAmount: '',
    Loan_Amount_Term: '',
    Credit_History: '',
    Property_Area: '',
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: (name.includes('Income') || name === 'LoanAmount' || name === 'Loan_Amount_Term') 
        ? (value === '' ? '' : Number(value)) 
        : value
    }));
  };

  // Progress logic
  const fields = Object.keys(formData);
  const completedFields = fields.filter(k => formData[k] !== '' && formData[k] !== null).length;
  const progressPercent = Math.round((completedFields / fields.length) * 100);

  // EMI logic
  const estimatedEMI = useMemo(() => {
    if (formData.LoanAmount > 0 && formData.Loan_Amount_Term > 0) {
      // Very simple approximation: (Principal * 1000) / Term (ignore interest for this mockup)
      return Math.round((formData.LoanAmount * 1000) / formData.Loan_Amount_Term);
    }
    return 0;
  }, [formData.LoanAmount, formData.Loan_Amount_Term]);

  const totalIncome = (Number(formData.ApplicantIncome) || 0) + (Number(formData.CoapplicantIncome) || 0);
  const ltiRatio = formData.LoanAmount ? (formData.LoanAmount / (totalIncome / 1000 + 1e-5)).toFixed(2) : '0.00';

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (completedFields < fields.length) {
      setError("Please complete all fields before assessing risk.");
      return;
    }
    
    if (formData.ApplicantIncome < 0 || formData.CoapplicantIncome < 0 || formData.LoanAmount <= 0) {
      setError("Income and loan amounts must be positive values.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
      const data = await response.json();
      setResult(data);
      
      // Scroll to result smoothly
      setTimeout(() => {
        document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      setError(err.message || "Failed to reach the prediction server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row gap-8 items-start relative pb-24">
      <div className="flex-1 w-full space-y-8">
        
        {/* Progress Bar */}
        <div className="bg-card shadow-card-light dark:shadow-card-dark-ambient rounded-2xl p-6 border-none">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-semibold text-textMain">{completedFields} of {fields.length} fields completed</span>
            <span className="text-xs font-bold text-accent">{progressPercent}%</span>
          </div>
          <div className="h-2 w-full bg-inputBg rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-accent to-secondaryAccent transition-all duration-500 ease-out" style={{ width: `${progressPercent}%` }}></div>
          </div>
        </div>

        <form id="assessment-form" onSubmit={handleSubmit} className="space-y-8">
          
          {/* Personal Details */}
          <section className="bg-card shadow-card-light dark:shadow-card-dark-ambient rounded-3xl p-6 sm:p-8 border-none">
            <h2 className="text-xl font-bold text-textMain mb-6 flex items-center gap-3">
              <div className="p-2 bg-accent/10 text-accent rounded-xl"><User size={20} /></div>
              Personal Details
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Gender</label>
                <PillToggle name="Gender" options={["Male", "Female"]} value={formData.Gender} onChange={handleChange} />
              </div>
              <div>
                <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Marital Status</label>
                <PillToggle name="Married" options={["Single", "Married"]} value={formData.Married} onChange={handleChange} />
              </div>
              <div>
                <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Dependents</label>
                <PillToggle name="Dependents" options={["0", "1", "2", "3+"]} value={formData.Dependents} onChange={handleChange} />
              </div>
              <div>
                <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Education</label>
                <PillToggle name="Education" options={["Graduate", "Not Graduate"]} value={formData.Education} onChange={handleChange} />
              </div>
              <div>
                <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Self Employed</label>
                <PillToggle name="Self_Employed" options={["No", "Yes"]} value={formData.Self_Employed} onChange={handleChange} />
              </div>
            </div>
          </section>

          {/* Financial Profile */}
          <section className="bg-card shadow-card-light dark:shadow-card-dark-ambient rounded-3xl p-6 sm:p-8 border-none">
            <h2 className="text-xl font-bold text-textMain mb-6 flex items-center gap-3">
              <div className="p-2 bg-accent/10 text-accent rounded-xl"><Wallet size={20} /></div>
              Financial Profile
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Applicant Monthly Income (₹)</label>
                <input type="number" name="ApplicantIncome" value={formData.ApplicantIncome} onChange={handleChange} min="0" placeholder="e.g. 4500" 
                  className="w-full bg-inputBg border border-inputBorder rounded-xl text-textMain p-4 focus:ring-0 focus:shadow-input-focus outline-none transition-shadow" />
              </div>
              <div>
                <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Coapplicant Monthly Income (₹)</label>
                <input type="number" name="CoapplicantIncome" value={formData.CoapplicantIncome} onChange={handleChange} min="0" placeholder="e.g. 1500" 
                  className="w-full bg-inputBg border border-inputBorder rounded-xl text-textMain p-4 focus:ring-0 focus:shadow-input-focus outline-none transition-shadow" />
              </div>
              <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-8 bg-black/5 dark:bg-white/5 p-6 rounded-2xl">
                <div>
                  <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Loan Amount (₹ in thousands)</label>
                  <input type="number" name="LoanAmount" value={formData.LoanAmount} onChange={handleChange} min="10" placeholder="e.g. 150" 
                    className="w-full bg-inputBg border border-inputBorder rounded-xl text-textMain p-4 focus:ring-0 focus:shadow-input-focus outline-none transition-shadow" />
                </div>
                <div>
                  <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Loan Term (Months)</label>
                  <div className="relative">
                    <select name="Loan_Amount_Term" value={formData.Loan_Amount_Term} onChange={handleChange} 
                      className="w-full bg-inputBg border border-inputBorder rounded-xl text-textMain p-4 focus:ring-0 focus:shadow-input-focus outline-none transition-shadow appearance-none">
                      <option value="" disabled>Select Term</option>
                      <option value={360}>360 (30 Years)</option>
                      <option value={240}>240 (20 Years)</option>
                      <option value={180}>180 (15 Years)</option>
                      <option value={120}>120 (10 Years)</option>
                      <option value={84}>84 (7 Years)</option>
                      <option value={60}>60 (5 Years)</option>
                      <option value={36}>36 (3 Years)</option>
                      <option value={12}>12 (1 Year)</option>
                    </select>
                  </div>
                </div>
                <div className="md:col-span-2 pt-2 text-center text-sm text-textMuted font-medium">
                  {estimatedEMI > 0 ? (
                    <span className="text-accent bg-accent/10 px-3 py-1 rounded-full">Estimated EMI: ~₹{estimatedEMI.toLocaleString('en-IN')}/mo</span>
                  ) : (
                    <span>Enter loan amount and term for EMI estimate</span>
                  )}
                </div>
              </div>
            </div>
          </section>

          {/* Credit & Property */}
          <section className="bg-card shadow-card-light dark:shadow-card-dark-ambient rounded-3xl p-6 sm:p-8 border-none">
            <h2 className="text-xl font-bold text-textMain mb-6 flex items-center gap-3">
              <div className="p-2 bg-accent/10 text-accent rounded-xl"><ShieldCheck size={20} /></div>
              Credit & Property
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
              <div className="space-y-8">
                <div>
                  <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Credit History Met</label>
                  <PillToggle name="Credit_History" options={["Yes (Good)", "No (Poor/None)"]} value={formData.Credit_History} onChange={handleChange} />
                  <p className="mt-3 text-xs text-textMuted italic">Whether the applicant has a track record of repaying debts on time.</p>
                </div>
                <div>
                  <label className="block text-xs uppercase tracking-wider text-textMuted font-bold mb-3">Property Area</label>
                  <PillToggle name="Property_Area" options={["Semiurban", "Urban", "Rural"]} value={formData.Property_Area} onChange={handleChange} />
                </div>
              </div>
              <div className="bg-secondaryAccent/10 border border-secondaryAccent/20 p-5 rounded-2xl h-full flex flex-col justify-center">
                <span className="text-lg mb-2">💡 Quick Tip</span>
                <p className="text-sm text-textMain font-medium leading-relaxed">
                  Credit history is historically the strongest predictor of loan approval. Ensure this field is highly accurate.
                </p>
              </div>
            </div>
          </section>
        </form>

        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-xl text-red-500 text-sm font-medium animate-in fade-in">
            {error}
          </div>
        )}

        {/* Results Panel Injection */}
        {result && (
          <div id="results-panel">
            <ResultsPanel data={result} formData={formData} />
          </div>
        )}

      </div>

      {/* Sticky Summary Sidebar (Desktop) */}
      <aside className="hidden lg:block w-80 sticky top-8 bg-card shadow-card-light dark:shadow-card-dark-ambient rounded-3xl p-6 border-none">
        <h3 className="text-sm uppercase tracking-wider text-textMuted font-bold mb-6">Live Summary</h3>
        <div className="space-y-6">
          <div>
            <div className="text-xs text-textMuted mb-1">Total Household Income</div>
            <div className="text-2xl font-bold font-display">₹{totalIncome.toLocaleString('en-IN')}</div>
          </div>
          <div>
            <div className="text-xs text-textMuted mb-1">Loan-to-Income Ratio</div>
            <div className="text-2xl font-bold font-display">{ltiRatio}</div>
          </div>
          <div>
            <div className="text-xs text-textMuted mb-1">Estimated EMI</div>
            <div className="text-2xl font-bold font-display text-secondaryAccent">₹{estimatedEMI > 0 ? estimatedEMI.toLocaleString('en-IN') : '0'}</div>
          </div>
        </div>
      </aside>

      {/* Sticky Bottom CTA */}
      <div className="fixed bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-background via-background to-transparent z-50 pointer-events-none">
        <div className="max-w-6xl mx-auto flex justify-end">
          <button 
            type="submit" 
            form="assessment-form"
            disabled={loading || completedFields < fields.length}
            className="w-full sm:w-auto px-12 py-4 bg-gradient-to-r from-accent to-secondaryAccent text-white font-bold rounded-full shadow-btn-glow transition-transform active:scale-95 disabled:opacity-50 disabled:shadow-none flex items-center justify-center gap-2 pointer-events-auto"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : (
              <>
                <Check size={20} />
                Assess Risk
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
