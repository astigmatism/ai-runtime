# UI mock-ups

Four static states of the viewport, in the palette and vocabulary the page itself uses. They are
the design of record for the rebuild: `runtime/static/index.html`, `app.js` and `style.css` follow
them, and nothing here is loaded by the application.

| File | State |
| --- | --- |
| `1-map-and-list.html` | Steady state. A solo configuration serving, a paired one selected, so the map also shows what a switch would place; one configuration's alias editor is open. |
| `2-switching.html` | A switch mid-flight: phase stepper, admissions paused, `Selected` / `Previous` / `Blocked` markers. |
| `3-restoration.html` | Nothing serving: the restore card, the map dimmed to what the selected configuration asks for, refusals on the rows that would not accept a switch. |
| `4-phone.html` | Below 900px: the map above the list, GPU tiles two per row. |

Every name, model, GPU, context window and publication sink is the real `rosalina` payload from
`GET /api/status`, not invented copy.

## Viewing

Open a file in a browser, or render a PNG:

```sh
python3 -m http.server 8899 --directory docs/ui-mockups   # then open http://127.0.0.1:8899/
chromium --headless=new --no-sandbox --hide-scrollbars --window-size=1280,1800 \
  --screenshot=/tmp/mock.png docs/ui-mockups/1-map-and-list.html
```

`mock.css` carries inline-style-free state classes only, because the page is served under
`style-src 'self'` and the mock-ups are the CSS the rebuild starts from. The PNG files are the
rendered output of the HTML beside them; regenerate rather than editing a picture.

To see the real page with these states instead of a picture, run the dev preview harness — it boots
`runtime/web.py` against `runtime/static/` with a recorded status fixture:

```sh
python3 scripts/preview-ui.py --state switching --port 8899
```
