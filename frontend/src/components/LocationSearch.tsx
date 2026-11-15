import { useEffect, useRef, useState } from 'react';
import { Loader2, MapPin, Search } from 'lucide-react';
import { api, errorMessage } from '../services/api';
import type { Place } from '../services/types';
import { cx, inputClass } from './ui';

interface Props {
  label: string;
  value: Place | null;
  onChange: (p: Place | null) => void;
  placeholder?: string;
  dotColor: string;
  active?: boolean;
  onFocusPick?: () => void;
}

export function LocationSearch({ label, value, onChange, placeholder, dotColor, active, onFocusPick }: Props) {
  const [q, setQ] = useState(value?.name ?? '');
  const [results, setResults] = useState<Place[]>([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [hi, setHi] = useState(0);
  const timer = useRef<number | undefined>(undefined);
  const editing = useRef(false);

  useEffect(() => {
    if (!editing.current) setQ(value?.name ?? '');
  }, [value]);

  const runSearch = (text: string) => {
    window.clearTimeout(timer.current);
    if (text.trim().length < 2) {
      setResults([]);
      return;
    }
    timer.current = window.setTimeout(async () => {
      setBusy(true);
      try {
        const r = await api.search(text);
        setResults(r.results);
        setMsg(r.warning ?? (r.results.length ? null : 'No Hyderabad places match. Try another name or click the map.'));
        setHi(0);
      } catch (e) {
        setMsg(errorMessage(e));
      } finally {
        setBusy(false);
      }
    }, 350);
  };

  const pick = (p: Place) => {
    editing.current = false;
    onChange(p);
    setQ(p.name);
    setOpen(false);
  };

  return (
    <div className="relative">
      <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-granite-700 dark:text-granite-200">
        <span className="h-2.5 w-2.5 rounded-full ring-2 ring-white dark:ring-granite-900" style={{ background: dotColor }} />
        {label}
        {active && <span className="text-xs font-normal text-turmeric-600 dark:text-turmeric-400">click the map to set</span>}
      </label>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-granite-400" />
        <input
          className={cx(inputClass, 'pl-9 pr-9', active && 'border-turmeric-500')}
          value={q}
          placeholder={placeholder}
          aria-label={label}
          role="combobox"
          aria-expanded={open}
          onFocus={() => {
            setOpen(true);
            onFocusPick?.();
          }}
          onBlur={() => window.setTimeout(() => setOpen(false), 150)}
          onChange={(e) => {
            editing.current = true;
            setQ(e.target.value);
            setOpen(true);
            runSearch(e.target.value);
            if (!e.target.value) onChange(null);
          }}
          onKeyDown={(e) => {
            if (!open || !results.length) return;
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setHi((h) => Math.min(results.length - 1, h + 1));
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              setHi((h) => Math.max(0, h - 1));
            } else if (e.key === 'Enter') {
              e.preventDefault();
              pick(results[hi]);
            }
          }}
        />
        {busy && <Loader2 className="absolute right-3 top-2.5 h-4 w-4 animate-spin text-granite-400" />}
      </div>
      {open && (results.length > 0 || msg) && q.length >= 2 && (
        <ul role="listbox" className="absolute z-[1000] mt-1 max-h-72 w-full overflow-auto rounded-lg border border-granite-200 bg-white py-1 shadow-lg dark:border-granite-700 dark:bg-granite-800">
          {results.map((r, i) => (
            <li key={`${r.name}-${r.lat}-${r.lon}`}>
              <button
                role="option"
                aria-selected={i === hi}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(r)}
                className={cx('flex w-full items-start gap-2 px-3 py-2 text-left text-sm', i === hi ? 'bg-granite-100 dark:bg-granite-700' : 'hover:bg-granite-50 dark:hover:bg-granite-700/60')}
              >
                <MapPin className="mt-0.5 h-4 w-4 shrink-0 text-granite-400" />
                <span className="min-w-0">
                  <span className="block truncate text-granite-900 dark:text-white">{r.name}</span>
                  <span className="block truncate text-xs text-granite-500 dark:text-granite-400">
                    {r.source === 'landmark' ? `Built-in place, ${r.category ?? ''}` : r.display_name ?? 'OpenStreetMap result'}
                  </span>
                </span>
              </button>
            </li>
          ))}
          {msg && <li className="px-3 py-2 text-xs text-granite-500 dark:text-granite-400">{msg}</li>}
        </ul>
      )}
    </div>
  );
}
