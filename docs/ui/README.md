# The page, as it is

Four states of `runtime/static/`, photographed from `scripts/preview-ui.py` — a real controller, the
real registry, a real publication, and `runtime/system.py`'s fake Docker host, so every name, GPU,
context window, refusal and sink below is what `GET /api/status` actually said, not invented copy.

| File | State |
| --- | --- |
| `1-running.png` | Steady state. One model owns all four text cards and the vision card answers images on its own; the two names the operator saved are said in the header, the map, and the list. |
| `2-switching.png` | A drain in flight: phase stepper and elapsed time, admissions paused on every card, the target marked in the list, and the whole machine saying the same sentence about what is happening. |
| `3-restoration.png` | Nothing serving. The restore card with the retry the runtime already scheduled, the map empty because nothing is placed, the list still readable so the next choice can be made knowingly. |
| `4-phone.png` | Below 900px: the map above the list, GPU tiles two per row, a configuration's state and its editor's door on a line of their own. |

These four images are documentation, not an asset the page loads: nothing in `runtime/static/` refers
to this directory, and no HTML here is served.

## Looking at the page yourself

```sh
python3 scripts/preview-ui.py --port 8899                    # serving, with two aliases already saved
python3 scripts/preview-ui.py --port 8899 --state switching  # caught mid-drain
python3 scripts/preview-ui.py --port 8899 --state restoration
python3 scripts/preview-ui.py --port 8899 --profile qwen27b-q8-pair
```

Then open `http://127.0.0.1:8899/`. Clicking a configuration redraws the map as a promise; clicking
Switch really runs a transition against the fake Docker host. The `restoration` state edits the
payload after the fact and says so in the file's own docstring; the rest of it is the running
product. `scripts/` is never copied into the image, so this workbench costs the container nothing.

## What the design is for

One idea, stated once: the physical GPUs and who has them. Everything else on the page is either a
fact in the line under the title — requests in flight, where the catalog was published, when this was
read — or folded into the details at the bottom. The list on the right is the only other thing,
because it is the only other action: choose a configuration, or change what one is called.

The earlier proposals for this layout, with the CSS they were built from, are in the history of this
directory (`docs/ui-mockups/`, removed when the page was rebuilt). They are superseded by these
photographs, which is the only kind of design document that cannot drift.
