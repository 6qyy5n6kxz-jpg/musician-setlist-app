// Safari (iPad) can't iterate a ReadableStream with `for await`, which pdf.js uses to read a
// PDF's text (PDF → chart). Add the standard async iterator on top of getReader().
type AsyncIterableStream = ReadableStream & { [Symbol.asyncIterator]?: () => AsyncIterator<unknown>; values?: () => AsyncIterator<unknown> };

if (typeof ReadableStream !== "undefined" && !(ReadableStream.prototype as AsyncIterableStream)[Symbol.asyncIterator]) {
  const values = function (this: ReadableStream, options?: { preventCancel?: boolean }): AsyncIterableIterator<unknown> {
    const reader = this.getReader();
    return {
      async next() {
        try {
          const r = await reader.read();
          if (r.done) reader.releaseLock();
          return r as IteratorResult<unknown>;
        } catch (e) {
          reader.releaseLock();
          throw e;
        }
      },
      async return(value?: unknown) {
        if (!options?.preventCancel) await reader.cancel(value).catch(() => {});
        reader.releaseLock();
        return { done: true, value };
      },
      [Symbol.asyncIterator]() { return this; },
    };
  };
  Object.defineProperty(ReadableStream.prototype, "values", { value: values, writable: true, configurable: true });
  Object.defineProperty(ReadableStream.prototype, Symbol.asyncIterator, { value: values, writable: true, configurable: true });
}

export {};
