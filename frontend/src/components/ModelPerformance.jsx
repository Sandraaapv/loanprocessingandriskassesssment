import { useEffect, useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  LineChart, Line
} from 'recharts';
import { useTheme } from './ThemeProvider';
import { API_BASE_URL } from '../config';

export default function ModelPerformance() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const { isDark } = useTheme();

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(`${API_BASE_URL}/model-performance`);
        const result = await response.json();
        setData(result);
      } catch (err) {
        console.error("Error fetching performance data", err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-textMuted font-medium animate-pulse">Loading performance metrics...</div>;
  }

  if (!data) {
    return <div className="text-center py-12 text-red-500 font-medium">Failed to load performance metrics. Is the backend running?</div>;
  }

  const { metrics, feature_importances } = data;
  
  const gridColor = isDark ? "#1F1F1F" : "#E5E0D8";
  const axisColor = "#8B93A7";
  const tooltipBg = isDark ? "#1A1A1A" : "#FFFFFF";
  const tooltipBorder = isDark ? "#262626" : "#E5E0D8";
  const tooltipColor = isDark ? "#FFFFFF" : "#111827";

  return (
    <div className="space-y-8 animate-in fade-in duration-500 pb-24">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {Object.entries(metrics).map(([key, value]) => (
          <div key={key} className="bg-card shadow-card-light dark:shadow-card-dark-ambient border-none rounded-3xl p-6 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-16 h-16 bg-accent opacity-0 group-hover:opacity-10 transition-opacity rounded-bl-full"></div>
            <div className="text-[10px] uppercase tracking-wider text-textMuted font-bold mb-2">{key.replace('_', ' ')}</div>
            <div className={`text-4xl font-extrabold font-display ${isDark ? 'text-gradient' : 'text-accent'}`}>{value}%</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Feature Importance Chart */}
        <div className="bg-card shadow-card-light dark:shadow-card-dark-ambient border-none rounded-3xl p-8">
          <h3 className="text-xl font-bold text-textMain mb-8 font-display">Feature Importance</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={feature_importances} margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} horizontal={true} vertical={false} />
                <XAxis type="number" stroke={axisColor} fontSize={12} tickLine={false} axisLine={false} />
                <YAxis dataKey="feature" type="category" stroke={axisColor} fontSize={12} tickLine={false} axisLine={false} width={120} />
                <Tooltip 
                  cursor={{fill: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)'}}
                  contentStyle={{ backgroundColor: tooltipBg, borderColor: tooltipBorder, color: tooltipColor, borderRadius: '12px', padding: '12px' }}
                  itemStyle={{ color: '#FF5A4E', fontWeight: 'bold' }}
                />
                <Bar dataKey="importance" fill="#FF5A4E" radius={[0, 8, 8, 0]} barSize={24} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ROC Curve */}
        <div className="bg-card shadow-card-light dark:shadow-card-dark-ambient border-none rounded-3xl p-8">
          <h3 className="text-xl font-bold text-textMain mb-8 font-display">ROC Curve (Simulated)</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={[
                { fpr: 0, tpr: 0 },
                { fpr: 0.1, tpr: 0.4 },
                { fpr: 0.2, tpr: 0.7 },
                { fpr: 0.4, tpr: 0.85 },
                { fpr: 0.8, tpr: 0.95 },
                { fpr: 1, tpr: 1 },
              ]} margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke={axisColor} fontSize={12} tickLine={false} />
                <YAxis type="number" domain={[0, 1]} stroke={axisColor} fontSize={12} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: tooltipBg, borderColor: tooltipBorder, color: tooltipColor, borderRadius: '12px', padding: '12px' }}
                />
                <Line type="monotone" dataKey="tpr" stroke="#FF5A4E" strokeWidth={4} dot={false} activeDot={{r: 6, fill: '#8B5CF6', stroke: 'none'}} />
                <Line type="linear" dataKey="fpr" stroke={axisColor} strokeDasharray="5 5" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
