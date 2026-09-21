// Base URL for backend API calls
// Can be overridden in production using VITE_API_URL environment variable
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
