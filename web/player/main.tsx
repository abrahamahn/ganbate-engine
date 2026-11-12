import { createRoot } from 'react-dom/client';
import { App } from './App.tsx';
import './player.css';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Missing #root mount element');
}

createRoot(container).render(<App />);
