import { Radio, FlaskConical } from 'lucide-react';
import { Chip } from './ui';

/** Always-visible label so simulated traffic is never mistaken for live data. */
export function TrafficModeBadge({ mode, label }: { mode?: string | null; label?: string }) {
  if (mode === 'live') {
    return (
      <Chip tone="live">
        <Radio className="h-3.5 w-3.5" /> {label ?? 'Live traffic (TomTom)'}
      </Chip>
    );
  }
  return (
    <Chip tone="sim">
      <FlaskConical className="h-3.5 w-3.5" /> Simulated Traffic – Not Live Traffic
    </Chip>
  );
}
