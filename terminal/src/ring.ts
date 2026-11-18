/** A fixed-size buffer that writes over its oldest item when full. Iterates oldest first. */
export class Ring<T> {
  private readonly items: (T | undefined)[];
  private start = 0;
  private size = 0;

  constructor(readonly capacity: number) {
    this.items = new Array<T | undefined>(capacity);
  }

  get length(): number {
    return this.size;
  }

  push(item: T): void {
    const end = (this.start + this.size) % this.capacity;
    this.items[end] = item;
    if (this.size < this.capacity) this.size++;
    else this.start = (this.start + 1) % this.capacity;
  }

  newest(): T | undefined {
    return this.size === 0 ? undefined : this.at(this.size - 1);
  }

  clear(): void {
    this.items.fill(undefined);
    this.start = 0;
    this.size = 0;
  }

  *[Symbol.iterator](): Generator<T> {
    for (let i = 0; i < this.size; i++) yield this.at(i);
  }

  *newestFirst(): Generator<T> {
    for (let i = this.size - 1; i >= 0; i--) yield this.at(i);
  }

  private at(i: number): T {
    return this.items[(this.start + i) % this.capacity] as T;
  }
}
