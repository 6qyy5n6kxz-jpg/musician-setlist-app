// Attach the CSRF token to every POST form and same-origin fetch on the page.
(function () {
  var meta = document.querySelector('meta[name="csrf-token"]');
  if (!meta) return;
  var token = meta.getAttribute("content");

  function addToken(form) {
    if ((form.getAttribute("method") || "get").toLowerCase() !== "post") return;
    if (form.querySelector('input[name="csrf_token"]')) return;
    var input = document.createElement("input");
    input.type = "hidden";
    input.name = "csrf_token";
    input.value = token;
    form.appendChild(input);
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("form").forEach(addToken);
  });
  // Forms added after page load
  document.addEventListener("submit", function (e) { addToken(e.target); }, true);
  var nativeSubmit = HTMLFormElement.prototype.submit;
  HTMLFormElement.prototype.submit = function () {
    addToken(this);
    return nativeSubmit.call(this);
  };

  var nativeFetch = window.fetch;
  window.fetch = function (input, init) {
    init = init || {};
    var method = (init.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
    var url = new URL(input instanceof Request ? input.url : input, window.location.href);
    if (method !== "GET" && method !== "HEAD" && url.origin === window.location.origin) {
      var headers = new Headers(init.headers || (input instanceof Request ? input.headers : undefined));
      headers.set("X-CSRFToken", token);
      init.headers = headers;
    }
    return nativeFetch.call(this, input, init);
  };
})();
