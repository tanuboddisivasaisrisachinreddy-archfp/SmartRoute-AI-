import { Route, Routes } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import Overview from './pages/Overview';
import RouteOptimizer from './pages/RouteOptimizer';
import TrafficSimulation from './pages/TrafficSimulation';
import Training from './pages/Training';
import Comparison from './pages/Comparison';
import Analytics from './pages/Analytics';
import Datasets from './pages/Datasets';
import About from './pages/About';
import { Empty } from './components/ui';
import { Link } from 'react-router-dom';

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Overview />} />
        <Route path="route" element={<RouteOptimizer />} />
        <Route path="traffic" element={<TrafficSimulation />} />
        <Route path="training" element={<Training />} />
        <Route path="comparison" element={<Comparison />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="about" element={<About />} />
        <Route
          path="*"
          element={
            <Empty title="Page not found" action={<Link className="text-turmeric-600 underline" to="/">Go to the overview</Link>}>
              This address does not match any page in SmartRoute AI.
            </Empty>
          }
        />
      </Route>
    </Routes>
  );
}
