# The page, as it is

Four states of `runtime/static/`, photographed from `scripts/preview-ui.py` — a real controller, the
real registry, a real publication, and `runtime/system.py`'s fake Docker host, so every name, GPU,
context window, refusal and sink below is what `GET /api/status` actually said, not invented copy.

| File | State |
| --- | --- |
| `1-running.png` | Steady state. The configuration in service names one model, and it owns all four text cards while the vision card answers images for it; the name the operator saved for that model is said in the header, on its card, in the map and in the list. Add `--profile qwen27b-q8-pair` and the same page draws two model cards, two saved names, and the split of the cards between them. |
| `2-switching.png` | A drain in flight: phase stepper and elapsed time under the configuration being changed, admissions paused on every card, the target marked in the list, and the whole machine saying the same sentence about what is happening — the operation's own line ends with `Stops: <model id>, …`, the models the target does not name. |
| `3-restoration.png` | Nothing serving. The restore card with the retry the runtime already scheduled, the map empty because nothing is placed, the list still readable so the next choice can be made knowingly. |
| `4-phone.png` | Below 900px: the map above the list, GPU tiles two per row, and a model's state and its **Rename** door on a line of their own. |

These four images are documentation, not an asset the page loads: nothing in `runtime/static/` refers
to this directory, and no HTML here is served.

## Looking at the page yourself

```sh
python3 scripts/preview-ui.py --port 8899                    # serving, each loaded model already named
python3 scripts/preview-ui.py --port 8899 --state switching  # caught mid-drain
python3 scripts/preview-ui.py --port 8899 --state restoration
python3 scripts/preview-ui.py --port 8899 --profile qwen27b-q8-pair
```

Then open `http://127.0.0.1:8899/`. Clicking a configuration redraws the map as a promise; clicking
Switch really runs a transition against the fake Docker host — from the solo configuration to
`qwen27b-q8-pair` that is a switch that stops the model in service, so the confirmation and the
operation line both name it. The `restoration` state edits the payload after the fact and says so in
the file's own docstring; the rest of it is the running product. `scripts/` is never copied into the
image, so this workbench costs the container nothing.

## What the page draws

The machine is drawn as its models. One card per model the selected configuration names — a machine
that loads two models has two cards, and neither is a footnote to the other — each with the name this
machine publishes for it, its context window, the cards it holds, and the service IDs a client sends
to reach it. A GPU tile is labelled with a service ID rather than with a model id, because that is
what an operator types and what fits on a tile; a tile that says `nighttime` is a tile about the model
that answers `nighttime` on this machine right now. The vision card is drawn apart, because a shared
card is not a model.

The list on the right is headed by a configuration's **id** — `qwen27b-q8-pair` — and not by a name,
because a name belongs to a model and a configuration is the decision about which models share which
cards. Under the id: the model ids it names, then one line per model with its name, the cards it would
run on, and its own **Rename**. Two models means two names, and one of them is never a spare for the
other; a name saved here is the name that model carries in every configuration that loads it.

Selecting a configuration that is not the live one redraws the map as a promise and writes the cost
under it: which model would take which cards, which cards would be left with nothing on them, and —
in the same shape as the models the switch loads, in the colour they have while they run — the models
that **stop**, because a switch runs the models the target names and nothing else. Under each stopping
model its service IDs are said per ID, because an ID another loaded model also answers does not stop
being answered, and the confirmation before the drain repeats only the IDs nothing in the target would
answer.

## What the design is for

One idea, stated once: the physical GPUs and which model has them. Everything else on the page is
either a fact in the line under the title — requests in flight, where the catalog was published, when
this was read — or folded into the details at the bottom. The list on the right is the only other
thing, because it is the only other action: choose a configuration, or change what one of its models
is called.

The earlier proposals for this layout, with the CSS they were built from, are in the history of this
directory (`docs/ui-mockups/`, removed when the page was rebuilt). They are superseded by these
photographs, which is the only kind of design document that cannot drift.

## The two columns are not negotiable

The left column is this machine as it runs; the right column is the choice. They are wrapped in the
markup, and no panel may be added beside them, because three children auto-flowing into two columns
place whatever comes third wherever a gap happens to be. Two machines on one revision used to draw
two different pages for exactly that reason: a host holding a durable operation receipt spent its
right column on the progress box — Check, Prepare, Drain, Load, Verify — and pushed the
configuration list under the map, while a host with no receipt to show kept the list on the right.
The one control an operator looks for moved sides depending on whether the machine had ever done
anything, which is worse than an ugly control in a fixed place.

The progress box belongs under the configuration whose change it is reporting, and it is about now:
a run in flight, a failure, a needs-attention. A run that finished goes quiet five minutes after it
stopped, since the receipt is one `GET /api/operations/<id>` away and the revision it ran is in the
runtime details and in the published catalog. A restoration from three days ago is history, and
history does not take room away from the choice.
