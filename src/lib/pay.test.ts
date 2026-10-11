import { describe, expect, it } from "vitest";
import { availableMethods, cleanHandle, payLink } from "./pay";

describe("tip links", () => {
  it("cleans whatever handle format people paste", () => {
    expect(cleanHandle("venmo", "@Devin-Frank-5")).toBe("Devin-Frank-5");
    expect(cleanHandle("venmo", "https://venmo.com/u/Devin-Frank-5")).toBe("Devin-Frank-5");
    expect(cleanHandle("cashapp", "$devinfrank")).toBe("devinfrank");
    expect(cleanHandle("cashapp", "https://cash.app/$devinfrank")).toBe("devinfrank");
    expect(cleanHandle("paypal", "paypal.me/devinfrank/10")).toBe("devinfrank");
  });
  it("builds links with the amount and note filled in", () => {
    expect(payLink("venmo", "@acop", 10, "Request: Wagon Wheel")).toBe("https://venmo.com/acop?txn=pay&amount=10&note=Request%3A+Wagon+Wheel");
    expect(payLink("cashapp", "$acop", 5)).toBe("https://cash.app/$acop/5");
    expect(payLink("paypal", "acop", 20)).toBe("https://paypal.me/acop/20USD");
    expect(payLink("cashapp", "acop")).toBe("https://cash.app/$acop");
  });
  it("lists only the methods that have a handle", () => {
    expect(availableMethods({ venmo: "@a", cashapp: "", paypal: null })).toEqual(["venmo"]);
  });
});
