// "Send to Stage" — runs inside Safari on the page you're viewing (Shortcuts: Run JavaScript on Web Page).
// Ultimate Guitar pages carry the chart as JSON in .js-store; other sites fall back to selected text or <pre>.
(function () {
  var APP = "https://stage.achangeofplansmusic.com/#/import?";
  var text = "";
  try {
    var store = document.querySelector(".js-store");
    if (store) {
      var data = JSON.parse(store.getAttribute("data-content")).store.page.data;
      var tab = data.tab || {};
      var view = data.tab_view || {};
      var content = (view.wiki_tab || {}).content;
      if (content) {
        var head = [];
        if (tab.song_name) head.push("{title: " + tab.song_name + "}");
        if (tab.artist_name) head.push("{artist: " + tab.artist_name + "}");
        if (tab.tonality_name) head.push("{key: " + tab.tonality_name + "}");
        if (view.meta && view.meta.capo) head.push("{capo: " + view.meta.capo + "}");
        text = head.join("\n") + "\n\n" + content;
      }
    }
  } catch (e) {}
  if (!text) {
    var sel = String(window.getSelection() || "");
    if (sel.trim()) text = "{title: " + document.title.split(/[|–-]/)[0].trim() + "}\n\n" + sel;
  }
  if (!text) {
    var pre = document.querySelector("pre");
    if (pre && pre.innerText.trim()) text = "{title: " + document.title.split(/[|–-]/)[0].trim() + "}\n\n" + pre.innerText;
  }
  if (!text) {
    completion(APP + "error=" + encodeURIComponent("No chart found on that page. Open the chart itself (or select its text) and share again."));
    return;
  }
  completion(APP + "text=" + encodeURIComponent(text));
})();
