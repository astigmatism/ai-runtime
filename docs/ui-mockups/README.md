# The card power state, as proposed

**Superseded, and kept.** The feature below is built: what it actually looks like is
[`5-power.png`](../ui/5-power.png) through [`9-power-phone.png`](../ui/9-power-phone.png) in
[`docs/ui/`](../ui/README.md), photographed from the running page. These four drawings stay because
they are the proposal the implementation was argued from, and because two of the six driver states on
the sheet below — a half-readable card, and no reading at all — are still easier to compare side by
side here than one at a time in a live preview.

Four mock-ups of the one feature the AI Runtime page did not have yet: **choosing an undervoltage
setting for a GPU and applying it**, which today lives only in
`/home/astigmatism/ops/scripts/gpu-undervolt` on Rosalina. Unlike [the photographs in
`docs/ui/`](../ui/README.md) — which are pictures of the built page — these were proposals for an
unbuilt feature, so they are HTML rather than a workbench run, and the copy inside them is the copy
the implementation was meant to ship. The existing page around them is copied for context, not
proposed again.

The rung names, watts, clock caps, offsets, defaults and measured evidence are taken from
`gpu-undervolt` unchanged: `safe` / `sensible` / `assertive` at 15 MHz offset steps, `sensible` the
default for the RTX 3090 and RTX 3080 Ti, `safe` the default for the RTX 4080 SUPER, no plan at all
for the RTX 3080 and the RTX 4080. Nothing is invented.

| File | State |
| --- | --- |
| [`power-1-tiles.html`](power-1-tiles.html) · [`.png`](power-1-tiles.png) | The **Physical GPUs** board with one power line and one **Power** button added to every tile, and below it a state sheet of the six readings the driver can give back: a rung applied, nothing applied, hand-set values, a card nobody has measured, a half-readable card, and no reading at all. The bold word is always what the driver reports *now*, never what the page last asked for. |
| [`power-2-chooser.html`](power-2-chooser.html) · [`.png`](power-2-chooser.png) | The chooser, opened under the card it belongs to: the values read from the driver and how fresh they are, a sentence about what cannot be verified on this driver, the rungs with this card's default marked and the evidence quoted, **Stock** as a rung-like row, the folded **Values by hand** block, the volatile-settings sentence, **Re-apply**, and the record of the last writes. |
| [`power-3-refusals.html`](power-3-refusals.html) · [`.png`](power-3-refusals.png) | The four states that must never look like success: refused because the model on that card is unhealthy (with the explicit *apply anyway*), written but read back wrong, a card with no plan, and a machine held by a switch. |
| [`power-4-phone.html`](power-4-phone.html) · [`.png`](power-4-phone.png) | 430 px: the map above the chooser, tiles two up, the rung rows still one line each, the buttons full width. The tile drops the watt figure and keeps the rung name. |

## What is proposed and what is only context

Proposed: the power line and **Power** button on a GPU tile, the chooser panel that opens under the
board inside the same column, the last sentence of the footer, and the state sheet panel (documentation,
not a part of the page). Context copied from the running product: everything else.

`mock.css` is `runtime/static/style.css` with one appended block, marked, holding every rule the
proposal adds. It keeps the page's own constraint — no inline styles — because the real page is served
under `style-src 'self'`, and a design that breaks that constraint is not implementable.

## Looking at them yourself

```sh
chromium --headless=new --no-sandbox --hide-scrollbars --window-size=1440,3600 \
  --screenshot=/tmp/power.png file://$PWD/docs/ui-mockups/power-2-chooser.html
python3 scripts/mockup-shot.py /tmp/power.png docs/ui-mockups/power-2-chooser.png
```

`scripts/mockup-shot.py` crops the uniform background below the page so the picture is the page's own
height, the same shape as the photographs in `docs/ui/`. Anything narrower than 900 px — the phone
picture is 430 — uses `--window-size=430,4400` instead.

## When this directory stops being useful

When the feature ships, `scripts/preview-ui.py` grows a power state and `docs/ui/` gains photographs of
the real panel, at which point this README records that it was superseded, as it did once before.
