import { useEffect, useMemo, useState } from "react";
import type { LoadStatus, WorkloadInfo } from "../contracts/control";
import { controlClient } from "../control";
import { launchSpec, statusLine, type LaunchForm } from "../launcher";

const POLL_MS = 1000;

export function Launcher({ control }: { control: string }) {
  const client = useMemo(() => controlClient(control), [control]);
  const [workloads, setWorkloads] = useState<WorkloadInfo[]>([]);
  const [status, setStatus] = useState<LoadStatus | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [form, setForm] = useState<LaunchForm>({
    workload: "",
    arrival: "closed",
    concurrency: 8,
    ratePerS: 2,
    maxTokens: 128,
    requests: 64,
  });

  const act = (call: Promise<LoadStatus>) =>
    call.then(
      (s) => (setStatus(s), setProblem(null)),
      (e: Error) => setProblem(e.message),
    );

  useEffect(() => {
    client.workloads().then(
      (found) => {
        setWorkloads(found);
        setForm((f) => (f.workload || !found[0] ? f : { ...f, workload: found[0].name }));
      },
      (e: Error) => setProblem(e.message),
    );
    void act(client.status());
  }, [client]);

  const running = status?.state === "running";
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => void act(client.status()), POLL_MS);
    return () => clearInterval(timer);
  }, [client, running]);

  const most = workloads.find((w) => w.name === form.workload)?.requests ?? 1;
  const set = (change: Partial<LaunchForm>) => setForm((f) => ({ ...f, ...change }));
  const num = (name: keyof LaunchForm, min: number, max: number, step = 1) => (
    <input
      type="number"
      name={name}
      min={min}
      max={max}
      step={step}
      value={form[name] as number}
      onChange={(e) => set({ [name]: e.target.valueAsNumber })}
    />
  );

  return (
    <section className="panel side-panel launcher" aria-label="load launcher">
      <h2>
        load launcher <span className="dim">not a benchmark</span>
      </h2>
      <form
        className="controls"
        onSubmit={(e) => {
          e.preventDefault();
          void act(client.start(launchSpec(form)));
        }}
      >
        <label>
          workload
          <select name="workload" value={form.workload} onChange={(e) => set({ workload: e.target.value })}>
            {workloads.map((w) => (
              <option key={w.name} value={w.name}>
                {w.name} ({w.requests})
              </option>
            ))}
          </select>
        </label>
        <label>
          arrival
          <select name="arrival" value={form.arrival} onChange={(e) => set({ arrival: e.target.value as LaunchForm["arrival"] })}>
            <option value="closed">closed loop</option>
            <option value="open">open loop</option>
          </select>
        </label>
        {form.arrival === "closed" ? (
          <label>concurrency {num("concurrency", 1, 64)}</label>
        ) : (
          <label>rate per s {num("ratePerS", 0.1, 50, 0.1)}</label>
        )}
        <label>max_tokens {num("maxTokens", 1, 4096)}</label>
        <label>requests {num("requests", 1, most)}</label>
        <div className="buttons">
          <button type="submit" disabled={running || !form.workload}>
            fire load
          </button>
          <button type="button" disabled={!running} onClick={() => void act(client.stop())}>
            stop
          </button>
        </div>
      </form>
      <p className="line" data-load={status?.state ?? "unknown"}>
        {status ? statusLine(status) : "…"}
      </p>
      {problem && <p className="line status-error">{problem}</p>}
    </section>
  );
}
