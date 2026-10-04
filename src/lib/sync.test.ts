import { describe, expect, it } from "vitest";
import { toServerRow } from "./sync";

describe("toServerRow", () => {
  it("gives old and new rows the same keys (the API rejects mixed batches)", () => {
    const oldRow = { id: "1", title: "Old", artist: "", content: "", created_at: "2025-01-01", updated_at: "2025-01-01", deleted_at: null, dirty: 1 };
    const newRow = { id: "2", title: "New", karaoke: true, gear: { numa: "x" }, key_kendra: "G", instrument: "piano", created_at: "2026-10-04", updated_at: "2026-10-04", deleted_at: null, dirty: 1, owner_id: "u", server_updated_at: "z" };
    const a = toServerRow("songs", oldRow);
    const b = toServerRow("songs", newRow);
    expect(Object.keys(a).sort()).toEqual(Object.keys(b).sort());
    expect(a).toMatchObject({ karaoke: false, gear: {}, key_kendra: null, requestable: true });
    expect(b).toMatchObject({ karaoke: true, key_kendra: "G" });
    expect(b).not.toHaveProperty("dirty");
    expect(b).not.toHaveProperty("owner_id");
  });
});
