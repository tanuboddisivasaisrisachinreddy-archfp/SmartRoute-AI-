import { useEffect, useMemo, useRef } from 'react';
import { canvas, latLngBounds, layerGroup, polyline, type Polyline as LPolyline } from 'leaflet';
import { Circle, CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet';
import type { Incident, NetworkData } from '../services/types';
import { congestionColor } from '../components/ui';

export const HYDERABAD_CENTER: [number, number] = [17.405, 78.445];

export interface MapRoute {
  id: string;
  color: string;
  geometry: [number, number][];
  label: string;
  dashed?: boolean;
}

interface Props {
  routes?: MapRoute[];
  selected?: string | null;
  onSelectRoute?: (id: string) => void;
  source?: { lat: number; lon: number; name?: string } | null;
  destination?: { lat: number; lon: number; name?: string } | null;
  incidents?: Incident[];
  network?: NetworkData | null;
  congestion?: number[] | null;
  onMapClick?: (lat: number, lon: number) => void;
  fitKey?: string;
  dark?: boolean;
  className?: string;
  zoom?: number;
}

function ClickHandler({ onClick }: { onClick?: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onClick?.(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function FitToRoutes({ routes, fitKey, source, destination }: { routes: MapRoute[]; fitKey?: string; source?: Props['source']; destination?: Props['destination'] }) {
  const map = useMap();
  useEffect(() => {
    const pts: [number, number][] = routes.flatMap((r) => r.geometry);
    if (source) pts.push([source.lat, source.lon]);
    if (destination) pts.push([destination.lat, destination.lon]);
    if (pts.length >= 2) map.fitBounds(latLngBounds(pts), { padding: [48, 48], maxZoom: 15 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey]);
  return null;
}

/** Thousands of road segments drawn on a single canvas for speed; colours follow congestion. */
function NetworkOverlay({ network, congestion }: { network: NetworkData; congestion?: number[] | null }) {
  const map = useMap();
  const lines = useRef<LPolyline[]>([]);
  useEffect(() => {
    const renderer = canvas({ padding: 0.4 });
    const group = layerGroup();
    lines.current = network.segments.map((s) =>
      polyline(s.c, {
        renderer,
        color: '#8b9a9b',
        weight: s.k <= 1 ? 3.2 : s.k === 2 ? 2.4 : 1.5,
        opacity: 0.8,
        interactive: false,
      }).addTo(group),
    );
    group.addTo(map);
    return () => {
      group.remove();
      lines.current = [];
    };
  }, [network, map]);
  useEffect(() => {
    if (!congestion) return;
    network.segments.forEach((s, i) => {
      let c = 0;
      for (const id of s.ids) c = Math.max(c, congestion[id] ?? 0);
      lines.current[i]?.setStyle({ color: congestionColor(c) });
    });
  }, [congestion, network]);
  return null;
}

export default function HyderabadMap({
  routes = [], selected, onSelectRoute, source, destination, incidents = [], network, congestion, onMapClick, fitKey, dark, className, zoom = 12,
}: Props) {
  const ordered = useMemo(
    () => [...routes].sort((a, b) => (a.id === selected ? 1 : 0) - (b.id === selected ? 1 : 0)),
    [routes, selected],
  );
  const tiles = dark
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
  return (
    <MapContainer center={HYDERABAD_CENTER} zoom={zoom} minZoom={10} maxZoom={18} className={className ?? 'h-full w-full'} zoomControl>
      <TileLayer
        key={tiles}
        url={tiles}
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
      />
      {network && <NetworkOverlay network={network} congestion={congestion} />}
      <ClickHandler onClick={onMapClick} />
      <FitToRoutes routes={routes} fitKey={fitKey} source={source} destination={destination} />
      {ordered.map((r) => {
        const active = !selected || r.id === selected;
        return (
          <Polyline
            key={`${r.id}-${fitKey}`}
            positions={r.geometry}
            eventHandlers={{ click: () => onSelectRoute?.(r.id) }}
            pathOptions={{
              color: r.color,
              weight: r.id === selected ? 7 : 4.5,
              opacity: active ? 0.95 : 0.35,
              dashArray: r.dashed ? '6 8' : undefined,
              lineCap: 'round',
              lineJoin: 'round',
              className: r.dashed ? undefined : 'route-draw',
            }}
          >
            <Tooltip sticky>{r.label}</Tooltip>
          </Polyline>
        );
      })}
      {incidents.map((inc) => (
        <Circle
          key={inc.id}
          center={[inc.lat, inc.lon]}
          radius={inc.radius_m}
          pathOptions={{ color: '#c2362f', weight: 1.5, dashArray: '4 4', fillOpacity: 0.12 + 0.25 * inc.current_intensity }}
        >
          <Tooltip>
            {inc.label} · {inc.status} · intensity {inc.current_intensity.toFixed(2)}
          </Tooltip>
        </Circle>
      ))}
      {source && (
        <CircleMarker center={[source.lat, source.lon]} radius={8} pathOptions={{ color: '#132026', weight: 3, fillColor: '#ffffff', fillOpacity: 1 }}>
          <Tooltip permanent direction="top" offset={[0, -8]}>
            {source.name ?? 'Start'}
          </Tooltip>
        </CircleMarker>
      )}
      {destination && (
        <CircleMarker center={[destination.lat, destination.lon]} radius={8} pathOptions={{ color: '#ffffff', weight: 3, fillColor: '#e3a21a', fillOpacity: 1 }}>
          <Tooltip permanent direction="top" offset={[0, -8]}>
            {destination.name ?? 'Destination'}
          </Tooltip>
        </CircleMarker>
      )}
    </MapContainer>
  );
}
