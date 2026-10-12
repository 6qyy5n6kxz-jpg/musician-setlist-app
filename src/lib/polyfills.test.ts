import { expect, it } from "vitest";

it("lets Safari iterate a ReadableStream with for await (pdf.js text reading)", async () => {
  // Simulate Safari: no async iterator on ReadableStream
  const proto = ReadableStream.prototype as unknown as Record<symbol | string, unknown>;
  const native = proto[Symbol.asyncIterator];
  delete proto[Symbol.asyncIterator];
  delete proto.values;
  try {
    await import("./polyfills?safari");
    const stream = new ReadableStream({ start(c) { c.enqueue("a"); c.enqueue("b"); c.close(); } });
    const got: unknown[] = [];
    for await (const chunk of stream as unknown as AsyncIterable<unknown>) got.push(chunk);
    expect(got).toEqual(["a", "b"]);
  } finally {
    Object.defineProperty(proto, Symbol.asyncIterator, { value: native, writable: true, configurable: true });
  }
});
