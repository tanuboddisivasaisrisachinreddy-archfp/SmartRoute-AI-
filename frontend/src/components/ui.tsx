import { useEffect, type ReactNode } from 'react';
import { AlertTriangle, Inbox, Loader2, RefreshCw } from 'lucide-react';
import type { AlgorithmId } from '../services/types';

export const ALGO: Record<AlgorithmId, { label: string; color: string; short: string; blurb: string }> = {
  dijkstra: { label: 'Dijkstra', color: '#2563eb', short: 'DIJ', blurb: 'Classical shortest path on the departure-time traffic snapshot' },
  q_learning: { label: 'Q-Learning', color: '#16a34a', short: 'QL', blurb: 'Tabular reinforcement learning, one Q-table per destination' },
  dqn: { label: 'DQN', color: '#9333ea', short: 'DQN', blurb: 'Deep Q-Network scoring each outgoing road segment' },
};

export const fmt = (v: number | null | undefined, digits = 1, suffix = '') =>
  v === null || v === undefined || Number.isNaN(v) ? '–' : `${v.toFixed(digits)}${suffix}`;

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(' ');
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-2xl">
        <h1 className="text-[1.9rem] font-[750] leading-tight text-granite-900 dark:text-white">{title}</h1>
        {subtitle && <p className="mt-1.5 text-[0.95rem] leading-relaxed text-granite-500 dark:text-granite-300">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Panel({ title, action, children, className, padded = true }: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <section
      className={cx(
        'rounded-[14px] border border-granite-200/80 bg-white dark:border-granite-800 dark:bg-granite-900',
        padded && 'p-5',
        className,
      )}
    >
      {(title || action) && (
        <div className={cx('mb-4 flex items-center justify-between gap-3', !padded && 'px-5 pt-5')}>
          {title && <h2 className="text-[1.05rem] font-[650] text-granite-800 dark:text-granite-100">{title}</h2>}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, accent }: { label: string; value: ReactNode; hint?: ReactNode; accent?: string }) {
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-2 text-sm text-granite-500 dark:text-granite-300">
        {accent && <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: accent }} />}
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-1 font-display text-[1.65rem] font-[650] leading-none text-granite-900 dark:text-white">{value}</div>
      {hint && <div className="mt-1.5 text-xs text-granite-500 dark:text-granite-400">{hint}</div>}
    </div>
  );
}

type BtnVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export function Button({ children, onClick, variant = 'secondary', disabled, loading, type = 'button', icon, className, title }: {
  children?: ReactNode;
  onClick?: () => void;
  variant?: BtnVariant;
  disabled?: boolean;
  loading?: boolean;
  type?: 'button' | 'submit';
  icon?: ReactNode;
  className?: string;
  title?: string;
}) {
  const styles: Record<BtnVariant, string> = {
    primary: 'bg-turmeric-500 text-granite-950 hover:bg-turmeric-400 disabled:bg-turmeric-300/60',
    secondary: 'border border-granite-200 bg-white text-granite-800 hover:border-granite-300 hover:bg-granite-50 dark:border-granite-700 dark:bg-granite-800 dark:text-granite-100 dark:hover:bg-granite-700',
    ghost: 'text-granite-600 hover:bg-granite-100 dark:text-granite-300 dark:hover:bg-granite-800',
    danger: 'border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300',
  };
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled || loading}
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60',
        styles[variant],
        className,
      )}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function Chip({ children, tone = 'neutral', className }: { children: ReactNode; tone?: 'neutral' | 'sim' | 'live' | 'ok' | 'warn' | 'bad'; className?: string }) {
  const tones = {
    neutral: 'bg-granite-100 text-granite-700 dark:bg-granite-800 dark:text-granite-200',
    sim: 'bg-turmeric-300/40 text-turmeric-700 ring-1 ring-turmeric-400/60 dark:bg-turmeric-500/15 dark:text-turmeric-300',
    live: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
    ok: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300',
    warn: 'bg-amber-50 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
    bad: 'bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300',
  };
  return <span className={cx('inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium', tones[tone], className)}>{children}</span>;
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="grid auto-cols-fr grid-flow-col rounded-lg bg-granite-100 p-1 dark:bg-granite-800">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            'rounded-md px-2 py-1.5 text-sm transition-colors',
            value === o.value
              ? 'bg-white font-medium text-granite-900 shadow-[0_1px_2px_rgba(19,32,38,.12)] dark:bg-granite-600 dark:text-white'
              : 'text-granite-500 hover:text-granite-800 dark:text-granite-300 dark:hover:text-white',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-granite-700 dark:text-granite-200">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-granite-500 dark:text-granite-400">{hint}</span>}
    </label>
  );
}

export const inputClass =
  'w-full rounded-lg border border-granite-200 bg-white px-3 py-2 text-sm text-granite-900 placeholder:text-granite-400 focus:border-turmeric-500 focus:outline-none dark:border-granite-700 dark:bg-granite-800 dark:text-granite-100';

export function Loading({ label = 'Loading…', className }: { label?: string; className?: string }) {
  return (
    <div role="status" className={cx('flex items-center justify-center gap-2 py-10 text-sm text-granite-500 dark:text-granite-300', className)}>
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-3 rounded-xl border border-red-200 bg-red-50/70 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
      <div className="flex gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        <span>{message}</span>
      </div>
      {onRetry && (
        <Button onClick={onRetry} icon={<RefreshCw className="h-4 w-4" />}>
          Try again
        </Button>
      )}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-granite-300 px-6 py-10 text-center dark:border-granite-700">
      <Inbox className="h-6 w-6 text-granite-400" />
      <div className="font-medium text-granite-700 dark:text-granite-200">{title}</div>
      {children && <div className="max-w-md text-sm text-granite-500 dark:text-granite-400">{children}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function Toast({ message, tone = 'ok', onClose }: { message: string; tone?: 'ok' | 'bad'; onClose: () => void }) {
  useEffect(() => {
    const id = window.setTimeout(onClose, tone === 'ok' ? 3500 : 7000);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [message, tone]);
  return (
    <div
      role="status"
      className={cx(
        'fixed bottom-5 right-5 z-[1200] flex max-w-sm items-start gap-3 rounded-xl px-4 py-3 text-sm shadow-lg',
        tone === 'ok' ? 'bg-granite-900 text-white dark:bg-white dark:text-granite-900' : 'bg-red-700 text-white',
      )}
    >
      <span className="flex-1">{message}</span>
      <button onClick={onClose} className="opacity-70 hover:opacity-100" aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}

export function congestionColor(c: number): string {
  if (c < 0.2) return '#3f9d6b';
  if (c < 0.4) return '#c9b023';
  if (c < 0.6) return '#e07a1f';
  return '#c2362f';
}
