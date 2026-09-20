export default function Documentation() {
  return (
    <div className="bg-card shadow-card-light dark:shadow-card-dark-ambient border-none rounded-3xl p-8 lg:p-12 max-w-4xl animate-in fade-in duration-500 text-textMain space-y-12 mb-24">
      
      <section>
        <h2 className="text-3xl font-extrabold font-display mb-6">Overview</h2>
        <p className="leading-relaxed text-textMuted font-medium text-lg">
          The Loan Approval & Risk Assessment model is a decision-support tool designed to automatically evaluate credit risk based on an applicant's personal and financial profile. 
          Its primary business purpose is to streamline the initial stages of loan underwriting, identifying strong candidates immediately while flagging high-risk applications for manual review.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold font-display mb-6">Dataset Description</h2>
        <p className="leading-relaxed text-textMuted font-medium mb-6">
          The model was trained on the standard Loan Prediction dataset. It includes demographics, financial metrics, and historical credit behavior.
        </p>
        <div className="overflow-x-auto rounded-2xl border border-inputBorder">
          <table className="w-full text-sm text-left border-collapse">
            <thead className="text-[10px] uppercase tracking-wider text-textMuted bg-black/5 dark:bg-[#1a1a1a]">
              <tr>
                <th className="px-6 py-4 font-bold">Field</th>
                <th className="px-6 py-4 font-bold">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-inputBorder">
              <tr className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                <td className="px-6 py-4 font-bold text-textMain">ApplicantIncome</td>
                <td className="px-6 py-4 text-textMuted">Monthly income of the primary applicant</td>
              </tr>
              <tr className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                <td className="px-6 py-4 font-bold text-textMain">CoapplicantIncome</td>
                <td className="px-6 py-4 text-textMuted">Monthly income of the co-applicant</td>
              </tr>
              <tr className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                <td className="px-6 py-4 font-bold text-textMain">LoanAmount</td>
                <td className="px-6 py-4 text-textMuted">Loan amount requested in thousands</td>
              </tr>
              <tr className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                <td className="px-6 py-4 font-bold text-textMain">Credit_History</td>
                <td className="px-6 py-4 text-textMuted">Whether the applicant meets credit guidelines (Yes/No)</td>
              </tr>
              <tr className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                <td className="px-6 py-4 font-bold text-textMain">Property_Area</td>
                <td className="px-6 py-4 text-textMuted">Urban, Semiurban, or Rural</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold font-display mb-6">Methodology</h2>
        <p className="leading-relaxed text-textMuted font-medium mb-4">
          <strong className="text-textMain">Preprocessing:</strong> Missing values in numeric columns are imputed using the median, while categorical columns use the mode. Categorical variables are one-hot encoded (e.g., Property_Area) or binary mapped (e.g., Gender, Married). Two engineered features, Total_Income and Loan_to_Income_Ratio, are added to strengthen financial context.
        </p>
        <p className="leading-relaxed text-textMuted font-medium">
          <strong className="text-textMain">Modeling:</strong> We trained multiple models including Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting. The Random Forest model with hyperparameter tuning via GridSearchCV yielded the highest ROC-AUC score and was selected for production.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold font-display mb-6">API Reference</h2>
        
        <div className="bg-[#111827] dark:bg-[#1A1A1A] rounded-2xl p-6 mb-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <span className="bg-green-500/20 text-green-400 text-xs font-extrabold tracking-wider px-3 py-1.5 rounded-lg uppercase">POST</span>
            <code className="text-sm font-mono text-gray-200">/predict</code>
          </div>
          <p className="text-sm text-gray-400 mb-4">Accepts applicant data and returns an approval decision, risk score, and key influencing factors.</p>
          <pre className="bg-black/50 p-4 rounded-xl text-xs text-gray-300 overflow-x-auto font-mono">
            {`// Example Request
{
  "Gender": "Male",
  "ApplicantIncome": 5000,
  "LoanAmount": 150
  // ...other fields
}`}
          </pre>
        </div>

        <div className="bg-[#111827] dark:bg-[#1A1A1A] rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <span className="bg-blue-500/20 text-blue-400 text-xs font-extrabold tracking-wider px-3 py-1.5 rounded-lg uppercase">GET</span>
            <code className="text-sm font-mono text-gray-200">/model-performance</code>
          </div>
          <p className="text-sm text-gray-400">Returns model performance metrics and feature importances for rendering the dashboard charts.</p>
        </div>
      </section>

      <section className="bg-yellow-500/10 border border-yellow-500/20 rounded-2xl p-8">
        <h2 className="text-xl font-bold text-yellow-600 dark:text-yellow-500 mb-3 flex items-center gap-2">
          Limitations & Disclaimers
        </h2>
        <p className="text-sm text-yellow-700 dark:text-yellow-500/90 leading-relaxed font-medium">
          This system is intended for decision support only and should not be used as a fully automated sole decision-maker for credit approvals. 
          Models trained on historical approval data may inherit historical biases. Manual underwriter review is recommended for edge cases and rejections to ensure fair lending practices.
        </p>
      </section>

    </div>
  );
}
