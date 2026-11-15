import { describe, expect, it } from 'vitest';
import { AxiosError, AxiosHeaders, type InternalAxiosRequestConfig } from 'axios';
import { API_BASE, api, errorMessage, http } from './api';

function mockAdapter(responder: (cfg: InternalAxiosRequestConfig) => { status: number; data: unknown }) {
  const calls: InternalAxiosRequestConfig[] = [];
  http.defaults.adapter = async (config: InternalAxiosRequestConfig) => {
    calls.push(config);
    const { status, data } = responder(config);
    const res = { data, status, statusText: String(status), headers: {}, config };
    if (status >= 400) throw new AxiosError('fail', 'ERR_BAD_RESPONSE', config, null, res);
    return res;
  };
  return calls;
}

describe('frontend ↔ backend API contract', () => {
  it('targets the configured backend /api prefix', () => {
    expect(http.defaults.baseURL).toBe(`${API_BASE}/api`);
  });

  it('posts a compare request with all three algorithms and Hyderabad coordinates', async () => {
    const calls = mockAdapter(() => ({ status: 200, data: { routes: [], best_algorithm: null } }));
    await api.compare({
      source: { name: 'Hitech City', lat: 17.4504, lon: 78.381 },
      destination: { name: 'Charminar', lat: 17.3616, lon: 78.4747 },
      preference: 'fastest',
      scenario: 'peak',
      event: 'incident_on_route',
    });
    expect(calls[0].url).toBe('/routes/compare');
    expect(calls[0].method).toBe('post');
    const body = JSON.parse(calls[0].data as string);
    expect(body.algorithms).toEqual(['dijkstra', 'q_learning', 'dqn']);
    expect(body.source).toEqual({ lat: 17.4504, lon: 78.381, name: 'Hitech City' });
    expect(body.event).toBe('incident_on_route');
  });

  it('requests per-edge congestion for the traffic layer', async () => {
    const calls = mockAdapter(() => ({ status: 200, data: { mode: 'simulation' } }));
    await api.traffic(true);
    expect(calls[0].url).toBe('/traffic/current');
    expect(calls[0].params).toEqual({ include_edges: true });
  });

  it('turns FastAPI errors into readable messages', async () => {
    mockAdapter(() => ({ status: 422, data: { detail: [{ loc: ['body', 'source', 'lat'], msg: 'latitude outside the Hyderabad coverage area' }] } }));
    try {
      await api.search('x');
      throw new Error('should fail');
    } catch (e) {
      expect(errorMessage(e)).toBe('source.lat: latitude outside the Hyderabad coverage area');
    }
    mockAdapter(() => ({ status: 503, data: { detail: 'Road network not loaded' } }));
    await expect(api.health()).rejects.toBeTruthy();
    try {
      await api.health();
    } catch (e) {
      expect(errorMessage(e)).toBe('Road network not loaded');
    }
  });

  it('explains network failures', () => {
    const cfg = { headers: new AxiosHeaders() } as InternalAxiosRequestConfig;
    const err = new AxiosError('Network Error', 'ERR_NETWORK', cfg);
    expect(errorMessage(err)).toContain('Cannot reach the backend');
  });
});
