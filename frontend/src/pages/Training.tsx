import { useEffect, useState, type ChangeEvent } from 'react';
import { Play, Square } from 'lucide-react';
import { useAsync } from '../hooks/useAsync';
import { api, errorMessage } from '../services/api';
import { ALGO, Button, Chip, Empty, ErrorState, Field, Loading, PageHeader, Panel, Segmented, Stat, cx, fmt, inputClass } from '../components/ui';
import { TrendChart } from '../charts/Charts';
import type { CurvePoint, Preference, TrainingHistoryEntry } from '../services/types';

type Algo = 'q_learning' | 'dqn';

const DEFAULTS: Record<Algo, { episodes: number; learning_rate: number; discount_factor: number }> = {
  q_learning: { episodes: 300, learning_rate: 0.3, discount_factor: 1.0 },
  dqn: { episodes: 300, learning_rate: 0.0005, discount_factor: 0.995 },
};

function CurveCharts({ curve, color, showLoss }: { curve: CurvePoint[]; color: string; showLoss: boolean }) {
  return (
    <div className="grid gap-5 md:grid-cols-2">
      <div>
        <h3 className="mb-2 text-sm font-medium text-granite-600 dark:text-granite-300">Reward (rolling mean of 100 episodes)</h3>
        <TrendChart data={curve as unknown as Record<string, unknown>[]} xKey="episode" height={200}
          series={[{ key: 'avg_reward', name: 'Average reward', color }, { key: 'episode_reward', name: 'Episode reward', color: '#a3b1b1', dashed: true }]} />
      </div>
      <div>
        <h3 className="mb-2 text-sm font-medium text-granite-600 dark:text-granite-300">Completion rate and exploration</h3>
        <TrendChart data={curve as unknown as Record<string, unknown>[]} xKey="episode" height={200}
          series={[{ key: 'completion_rate', name: 'Reached destination', color: '#16a34a' }, { key: 'epsilon', name: 'Epsilon', color: '#e3a21a', dashed: true }]} />
      </div>
      {showLoss && (
        <div className="md:col-span-2">
          <h3 className="mb-2 text-sm font-medium text-granite-600 dark:text-granite-300">Training loss (Huber, rolling mean)</h3>
          <TrendChart data={curve.filter((c) => c.loss != null) as unknown as Record<string, unknown>[]} xKey="episode" height={180}
            series={[{ key: 'loss', name: 'Loss', color: '#9333ea' }]} />
        </div>
      )}
    </div>
  );
}

function HistoryBlock({ id, entry }: { id: Algo; entry: TrainingHistoryEntry | null }) {
  if (!entry) {
    return (
      <Empty title={`No saved ${ALGO[id].label} training`}>
        Run <code>python scripts/{id === 'dqn' ? 'train_dqn.py' : 'train_q_learning.py'}</code> or start a job above.
      </Empty>
    );
  }
  if (entry.error) return <ErrorState message={entry.error} />;
  return (
    <div className="space-y-4">
      <p className="text-sm text-granite-500">{entry.episodes.toLocaleString()} episodes logged.</p>
      {entry.curve.length > 1 && <CurveCharts curve={entry.curve} color={ALGO[id].color} showLoss={id === 'dqn'} />}
      {entry.summary && entry.summary.length > 0 && (
        <div className="max-h-72 overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-white text-left text-granite-500 dark:bg-granite-900">
              <tr><th className="py-1.5 font-medium">Destination</th><th className="font-medium">Preference</th><th className="font-medium">Episodes</th><th className="font-medium">Success (last 100)</th><th className="font-medium">Avg reward</th></tr>
            </thead>
            <tbody className="divide-y divide-granite-100 dark:divide-granite-800">
              {entry.summary.map((s) => (
                <tr key={`${s.goal}-${s.preference}`}>
                  <td className="py-1.5">{s.goal}</td><td>{s.preference}</td><td>{s.episodes}</td>
                  <td>{fmt(s.final_success_rate * 100, 0)}%</td><td>{fmt(s.final_avg_reward, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function Training() {
  const [algo, setAlgo] = useState<Algo>('q_learning');
  const [form, setForm] = useState({ ...DEFAULTS.q_learning, epsilon: 1, batch_size: 128, buffer_size: 40000, seed: 42 });
  const [prefs, setPrefs] = useState<Preference[]>(['fastest']);
  const [goals, setGoals] = useState('hitech-city,charminar,rgia-airport');
  const [startErr, setStartErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const status = useAsync(() => api.trainingStatus(), [], 2000);
  const history = useAsync(() => api.trainingHistory(), []);
  const state = status.data?.state;

  useEffect(() => {
    setForm((f) => ({ ...f, ...DEFAULTS[algo] }));
  }, [algo]);

  useEffect(() => {
    if (state === 'completed') void history.reload(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  const start = async () => {
    setStarting(true);
    setStartErr(null);
    try {
      await api.startTraining({ algorithm: algo, ...form, preferences: prefs, goals: algo === 'q_learning' ? goals || null : null });
      await status.reload(true);
    } catch (e) {
      setStartErr(errorMessage(e));
    } finally {
      setStarting(false);
    }
  };

  const p = status.data?.progress;
  const running = state === 'running' || state === 'starting';
  const num = (k: keyof typeof form) => (e: ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: Number(e.target.value) });

  return (
    <div>
      <PageHeader
        title="Reinforcement learning training"
        subtitle="Jobs run as a separate background process using the same scripts as offline training, so the API stays responsive. New checkpoints are loaded automatically when a job finishes."
      />
      <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
        <Panel title="New training job">
          <div className="space-y-4">
            <Segmented<Algo> label="Algorithm" value={algo} onChange={setAlgo} options={[{ value: 'q_learning', label: 'Q-Learning' }, { value: 'dqn', label: 'DQN' }]} />
            <div className="grid grid-cols-2 gap-3">
              <Field label={algo === 'q_learning' ? 'Episodes per Q-table' : 'Episodes'}><input type="number" min={10} className={inputClass} value={form.episodes} onChange={num('episodes')} /></Field>
              <Field label={algo === 'q_learning' ? 'Learning rate α' : 'Learning rate'}><input type="number" step="any" className={inputClass} value={form.learning_rate} onChange={num('learning_rate')} /></Field>
              <Field label="Discount γ"><input type="number" step="0.01" min={0} max={1} className={inputClass} value={form.discount_factor} onChange={num('discount_factor')} /></Field>
              <Field label="Initial ε"><input type="number" step="0.05" min={0} max={1} className={inputClass} value={form.epsilon} onChange={num('epsilon')} /></Field>
              {algo === 'dqn' && (
                <>
                  <Field label="Batch size"><input type="number" className={inputClass} value={form.batch_size} onChange={num('batch_size')} /></Field>
                  <Field label="Replay buffer"><input type="number" className={inputClass} value={form.buffer_size} onChange={num('buffer_size')} /></Field>
                </>
              )}
              <Field label="Random seed"><input type="number" className={inputClass} value={form.seed} onChange={num('seed')} /></Field>
            </div>
            {algo === 'q_learning' && (
              <>
                <Field label="Preferences">
                  <div className="flex gap-3 text-sm">
                    {(['fastest', 'balanced', 'shortest'] as Preference[]).map((pr) => (
                      <label key={pr} className="flex items-center gap-1.5">
                        <input type="checkbox" checked={prefs.includes(pr)} onChange={(e) => setPrefs(e.target.checked ? [...prefs, pr] : prefs.filter((x) => x !== pr))} />
                        {pr}
                      </label>
                    ))}
                  </div>
                </Field>
                <Field label="Destination landmarks" hint="Comma-separated ids; leave empty for all 19 (slow on a free server).">
                  <input className={inputClass} value={goals} onChange={(e) => setGoals(e.target.value)} />
                </Field>
              </>
            )}
            {startErr && <ErrorState message={startErr} />}
            <div className="flex gap-2">
              <Button variant="primary" className="flex-1" icon={<Play className="h-4 w-4" />} loading={starting} disabled={running || (algo === 'q_learning' && prefs.length === 0)} onClick={start}>
                Start training
              </Button>
              {running && <Button variant="danger" icon={<Square className="h-4 w-4" />} onClick={() => void api.stopTraining().then(() => status.reload(true))}>Stop</Button>}
            </div>
            <p className="text-xs leading-relaxed text-granite-500">
              For the full models use the command line on your Mac: <code>python scripts/train_q_learning.py</code> and <code>python scripts/train_dqn.py</code>. On Render free tier, keep jobs small: files written there are lost on redeploy.
            </p>
          </div>
        </Panel>

        <Panel
          title="Current job"
          action={state && <Chip tone={state === 'completed' ? 'ok' : state === 'failed' ? 'bad' : running ? 'sim' : 'neutral'}>{state}</Chip>}
        >
          {status.loading && !status.data ? (
            <Loading />
          ) : status.error ? (
            <ErrorState message={status.error} onRetry={() => void status.reload()} />
          ) : !p ? (
            <Empty title={state === 'idle' ? 'No job started in this session' : 'Waiting for the first progress report…'}>
              Start a job to watch reward, exploration rate, loss and completion rate update live.
            </Empty>
          ) : (
            <div className="space-y-5">
              <div>
                <div className="mb-1 flex justify-between text-sm"><span className="font-medium">{p.algorithm === 'dqn' ? 'DQN' : 'Q-Learning'}</span><span>{p.current_episode.toLocaleString()} / {p.total_episodes.toLocaleString()} episodes</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-granite-100 dark:bg-granite-800">
                  <div className={cx('h-full rounded-full transition-[width]', p.algorithm === 'dqn' ? 'bg-dqn' : 'bg-qlearn')} style={{ width: `${(100 * p.current_episode) / Math.max(1, p.total_episodes)}%` }} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-5 md:grid-cols-3 xl:grid-cols-6">
                <Stat label="Episode" value={p.current_episode.toLocaleString()} hint={`of ${p.total_episodes.toLocaleString()}`} />
                <Stat label="Average reward" value={fmt(p.average_reward, 1)} hint="last 100 episodes" />
                <Stat label="Episode reward" value={fmt(p.episode_reward, 1)} />
                <Stat label="Exploration ε" value={fmt(p.epsilon, 3)} />
                <Stat label="Training loss" value={p.loss != null ? fmt(p.loss, 4) : '–'} hint={p.algorithm === 'dqn' ? 'Huber loss' : 'tabular: no loss'} />
                <Stat label="Completion rate" value={`${fmt(p.completion_rate * 100, 0)}%`} hint="last 100 episodes" />
              </div>
              {p.curve.length > 1 && <CurveCharts curve={p.curve} color={ALGO[(p.algorithm as Algo) ?? 'q_learning'].color} showLoss={p.algorithm === 'dqn'} />}
              {status.data?.log_tail && <pre className="max-h-48 overflow-auto rounded-lg bg-granite-950 p-3 text-xs text-granite-200">{status.data.log_tail}</pre>}
            </div>
          )}
        </Panel>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Panel title="Saved Q-Learning model">{history.loading ? <Loading /> : history.error ? <ErrorState message={history.error} /> : <HistoryBlock id="q_learning" entry={history.data?.q_learning ?? null} />}</Panel>
        <Panel title="Saved DQN model">{history.loading ? <Loading /> : history.error ? <ErrorState message={history.error} /> : <HistoryBlock id="dqn" entry={history.data?.dqn ?? null} />}</Panel>
      </div>
    </div>
  );
}
