import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { useIsDark } from '../hooks/useTheme';

export interface SeriesDef {
  key: string;
  name: string;
  color: string;
  dashed?: boolean;
  axis?: 'left' | 'right';
}

function useChartTheme() {
  const dark = useIsDark();
  return {
    grid: dark ? '#2b3e45' : '#e3e8e7',
    axis: dark ? '#a3b1b1' : '#566a6d',
    tooltip: {
      background: dark ? '#1a2a31' : '#ffffff',
      border: `1px solid ${dark ? '#34454a' : '#c9d2d1'}`,
      borderRadius: 10,
      fontSize: 12,
      color: dark ? '#e3e8e7' : '#1a2a31',
    },
  };
}

const tickStyle = (color: string) => ({ fill: color, fontSize: 11 });

export function TrendChart({ data, xKey, series, height = 240, yLabel, rightLabel, xFormatter }: {
  data: Record<string, unknown>[];
  xKey: string;
  series: SeriesDef[];
  height?: number;
  yLabel?: string;
  rightLabel?: string;
  xFormatter?: (v: unknown) => string;
}) {
  const t = useChartTheme();
  const hasRight = series.some((s) => s.axis === 'right');
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: hasRight ? 8 : 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey={xKey} tick={tickStyle(t.axis)} tickFormatter={xFormatter} stroke={t.grid} minTickGap={24} />
        <YAxis yAxisId="left" tick={tickStyle(t.axis)} stroke={t.grid} width={48}
          label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: t.axis, fontSize: 11 } : undefined} />
        {hasRight && (
          <YAxis yAxisId="right" orientation="right" tick={tickStyle(t.axis)} stroke={t.grid} width={44}
            label={rightLabel ? { value: rightLabel, angle: 90, position: 'insideRight', fill: t.axis, fontSize: 11 } : undefined} />
        )}
        <Tooltip contentStyle={t.tooltip} labelFormatter={xFormatter ? (v) => xFormatter(v) : undefined} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s) => (
          <Line key={s.key} yAxisId={s.axis ?? 'left'} type="monotone" dataKey={s.key} name={s.name} stroke={s.color}
            strokeWidth={2} dot={false} strokeDasharray={s.dashed ? '5 5' : undefined} connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function AreaTrend({ data, xKey, series, height = 200, xFormatter, yMax }: {
  data: Record<string, unknown>[];
  xKey: string;
  series: SeriesDef[];
  height?: number;
  xFormatter?: (v: unknown) => string;
  yMax?: number;
}) {
  const t = useChartTheme();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey={xKey} tick={tickStyle(t.axis)} tickFormatter={xFormatter} stroke={t.grid} minTickGap={24} />
        <YAxis tick={tickStyle(t.axis)} stroke={t.grid} width={40} domain={[0, yMax ?? 'auto']} />
        <Tooltip contentStyle={t.tooltip} labelFormatter={xFormatter ? (v) => xFormatter(v) : undefined} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s) => (
          <Area key={s.key} type="monotone" dataKey={s.key} name={s.name} stroke={s.color} fill={s.color} fillOpacity={0.15}
            strokeWidth={2} isAnimationActive={false} />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function BarsChart({ data, xKey, series, height = 240, yLabel, layout = 'horizontal', cellColors }: {
  cellColors?: string[];
  data: Record<string, unknown>[];
  xKey: string;
  series: SeriesDef[];
  height?: number;
  yLabel?: string;
  layout?: 'horizontal' | 'vertical';
}) {
  const t = useChartTheme();
  const vertical = layout === 'vertical';
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={layout} margin={{ top: 8, right: 16, bottom: 4, left: vertical ? 8 : 0 }}>
        <CartesianGrid stroke={t.grid} vertical={vertical} horizontal={!vertical} />
        {vertical ? (
          <>
            <XAxis type="number" tick={tickStyle(t.axis)} stroke={t.grid} />
            <YAxis type="category" dataKey={xKey} tick={tickStyle(t.axis)} stroke={t.grid} width={150} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} tick={tickStyle(t.axis)} stroke={t.grid} />
            <YAxis tick={tickStyle(t.axis)} stroke={t.grid} width={48}
              label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: t.axis, fontSize: 11 } : undefined} />
          </>
        )}
        <Tooltip contentStyle={t.tooltip} cursor={{ fill: t.grid, opacity: 0.4 }} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s) => (
          <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color} radius={vertical ? [0, 4, 4, 0] : [4, 4, 0, 0]} isAnimationActive={false}>
            {cellColors && data.map((_, i) => <Cell key={i} fill={cellColors[i % cellColors.length]} />)}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
