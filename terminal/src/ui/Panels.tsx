import { stripModel } from "../strip";
import { CELL_WIDTH, drawStepChart, drawStrip } from "./draw";
import { StoreCanvas, type Draw } from "./StoreCanvas";
import type { TelemetryStore } from "../store";

const stepChart: Draw = (pen, store, width, height, palette) => drawStepChart(pen, store.seconds, width, height, palette);

const batchStrip: Draw = (pen, store, width, height, palette) =>
  drawStrip(pen, stripModel(store.steps, Math.floor(width / CELL_WIDTH)), width, height, palette);

export function StepChart({ store }: { store: TelemetryStore }) {
  return (
    <section className="panel chart" aria-label="step chart">
      <h2>
        step latency <span className="dim">one candle per second</span>
        <Key name="rise" label="rose" />
        <Key name="fall" label="fell" />
        <Key name="prefill" label="prefill" />
        <Key name="decode" label="decode" />
      </h2>
      <StoreCanvas store={store} draw={stepChart} />
    </section>
  );
}

export function BatchStrip({ store }: { store: TelemetryStore }) {
  return (
    <section className="panel strip" aria-label="batch strip">
      <h2>
        batch <span className="dim">one column per step, one lane per request</span>
        <Key name="prefill" label="prefill" />
        <Key name="decode" label="decode" />
      </h2>
      <StoreCanvas store={store} draw={batchStrip} />
    </section>
  );
}

function Key({ name, label }: { name: string; label: string }) {
  return (
    <span className="key">
      <span className={`swatch swatch-${name}`} />
      {label}
    </span>
  );
}
