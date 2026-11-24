import type { Frame } from "../contracts/telemetry";

type Event = Frame["events"][number];

/**
 * The trace played `times` times as one stream, for the long-session memory budget
 * (decision 0027). Only the first repeat keeps its `hello`, because a `hello` resets the
 * store. Each repeat shifts `seq` and every `t` by one period, and renames its requests.
 */
export function repeatTrace(frames: readonly Frame[], times: number): Frame[] {
  const first = frames[0]!.t;
  const period = frames.at(-1)!.t - first + (frames[1]!.t - first);
  const out: Frame[] = [];
  for (let k = 0; k < times; k++) {
    const shift = k * period;
    const rid = (id: string) => (k === 0 ? id : `${id}.${k}`);
    for (const frame of frames) {
      const events = frame.events.filter((e) => k === 0 || e.type !== "hello").map((e) => moved(e, shift, rid));
      out.push({ ...frame, seq: out.length, t: frame.t + shift, events });
    }
  }
  return out;
}

function moved(event: Event, shift: number, rid: (id: string) => string): Event {
  const e = { ...event, t: event.t + shift };
  if ("rid" in e) e.rid = rid(e.rid);
  if (e.type === "step") e.rids = e.rids.map(rid);
  if (e.type === "kv.snapshot") e.owners = e.owners.map(rid);
  return e;
}
