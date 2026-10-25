# The page, as it is

Nine states of `runtime/static/`, photographed from `scripts/preview-ui.py` — a real controller, the
real registry, a real publication, and `runtime/system.py`'s fake Docker host, so every name, GPU,
context window, refusal and sink below is what `GET /api/status` actually said, not invented copy.
The first four are the machine; the last five are one GPU card's power state, which lives inside the
same page.

| File | State |
| --- | --- |
| `1-running.png` | Steady state. The configuration in service names one model, and it owns all four text cards while the vision card answers images for it; the name the operator saved for that model is said in the header, on its card, in the map and in the list. Add `--profile qwen27b-q8-pair` and the same page draws two model cards, two saved names, and the split of the cards between them. |
| `2-switching.png` | A drain in flight: phase stepper and elapsed time under the configuration being changed, admissions paused on every card, the target marked in the list, and the whole machine saying the same sentence about what is happening — the operation's own line ends with `Stops: <model id>, …`, the models the target does not name. |
| `3-restoration.png` | Nothing serving. The restore card with the retry the runtime already scheduled, the map empty because nothing is placed, the list still readable so the next choice can be made knowingly. |
| `4-phone.png` | Below 900px: the map above the list, GPU tiles two per row, and a model's state and its **Rename** door on a line of their own. |
| `5-power.png` | Every card's power state as one word on its tile — a rung name, `stock`, `custom`, or `power unreadable` — with a **Power** door under it. `--power mixed` seeds the five states on purpose, including a card whose driver refuses the offset query. |
| `6-power-panel.png` | The panel opened under the map on a card sitting at `sensible`: the driver's own four readings, what the card is doing in a sentence, the measured rungs with **CARD DEFAULT** and **WHERE IT IS NOW** on the right ones, **Stock** as a reversible rung, the folded **Values by hand**, and the paragraph saying out loud that nothing is installed in the boot path. |
| `7-power-unmeasured.png` | A card this machine has never measured: no rungs to choose, so the panel offers only **Revert** and **Values by hand**, and says why nothing is pre-filled. It never inherits a 3090's numbers because another card happens to have them. |
| `8-power-refusal.png` | The health gate. The model on this card is unhealthy, so the panel refuses before the click instead of after, and the refusal carries the tick that means `--skip-health`: **Apply anyway — this engine is not what I am worried about**. The action button stays disabled until it is ticked. |
| `9-power-phone.png` | The same power lines at phone width: still one word per tile, still legible, with the panel opening under the map rather than beside it. |

These images are documentation, not an asset the page loads: nothing in `runtime/static/` refers
to this directory, and no HTML here is served.

## Looking at the page yourself

```sh
python3 scripts/preview-ui.py --port 8899                    # serving, each loaded model already named
python3 scripts/preview-ui.py --port 8899 --state switching  # caught mid-drain
python3 scripts/preview-ui.py --port 8899 --state restoration
python3 scripts/preview-ui.py --port 8899 --profile qwen27b-q8-pair
python3 scripts/preview-ui.py --port 8899 --power mixed      # every power state on a tile at once
python3 scripts/preview-ui.py --port 8899 --power gate       # an engine stopped: writes refuse
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

A GPU tile carries exactly one hardware fact: what the driver is doing with that card's power, read on
a slower cadence than the page polls, as one word — a measured rung's name, `stock`, `custom`, or
`power unreadable` when the driver will not say. **Power** under the tile opens the panel for that one
card, which is where the numbers live: the four readings, the rungs this machine measured for this
model of card, and the two actions. A rung is named from the values that can be read back, because a
GeForce reports no clock lock and no voltage — the observed SM clock is evidence, not proof, and the
panel says so rather than pretending to a verification it does not have. Nothing here is installed in
the boot path, and the panel says that too, in the same place the buttons are: a reboot returns the
card to stock on its own, which is the way back when a card disagrees with the rung it was given.

## What the design is for

One idea, stated once: the physical GPUs and which model has them. Everything else on the page is
either a fact in the line under the title — requests in flight, where the catalog was published, when
this was read — or folded into the details at the bottom. The list on the right is the only other
thing, because it is the only other action: choose a configuration, or change what one of its models
is called.

The earlier proposals for these layouts, with the CSS they were built from, are kept in
`docs/ui-mockups/`: the two-column page's proposals were removed when the page was rebuilt, and the
card-power panel's are still there, drawn before a line of the panel existed. All of them are
superseded by the photographs above, which is the only kind of design document that cannot drift.

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

The card power panel is in the left column, under the map it belongs to, for the same reason: what a
card is drawing right now is a fact about this machine as it runs, not a choice. It opens under the
map and never beside it, so the configuration list keeps the right column whichever tile was clicked.
