import { useCallback, useEffect, useRef, useState } from 'react';
import { errorMessage } from '../services/api';

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: (silent?: boolean) => Promise<void>;
  setData: (d: T | null) => void;
}

/** Fetch data on mount (and optionally poll).  Errors become readable strings. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = [], intervalMs?: number): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const reload = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const d = await fnRef.current();
      setData(d);
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    void reload();
    if (!intervalMs) return undefined;
    const id = window.setInterval(() => void reload(true), intervalMs);
    return () => window.clearInterval(id);
  }, [reload, intervalMs]);

  return { data, error, loading, reload, setData };
}
