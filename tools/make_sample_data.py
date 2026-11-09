#!/usr/bin/env python3
"""Generates SYNTHETIC sample screenplays (original text, not Kuku's) in two formats:

  sample_data/synthetic_show/          -> same layout as the supplied Kuku pack
  sample_data/synthetic_show_variant/  -> deliberately messy variant (HH:MM:SS clocks,
                                          un-bolded labels, missing headers / end times,
                                          BOM + CRLF, one untimed line)

Safe to publish; used by tests and by the public demo."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.timecode import format_timecode  # noqa: E402

D, A, S = "D", "A", "S"
EPISODES = [
    dict(n=1, logline="Heiress Mara Voss is humiliated at her own engagement party, then learns from a stranger that her dead father is alive.",
         cont="First episode.", scenes=[
        ("INT. GRAND BALLROOM \u2013 NIGHT", [
            (A, "Mara Voss, in a midnight-blue gown, freezes at the top of the staircase as the orchestra falls silent."),
            (D, "MARA", "Shaken", "Daniel? Tell me this is a joke."),
            (D, "DANIEL", "Cold, Dismissive", "The engagement is over, Mara. Sign the papers tonight."),
            (D, "MARA", "Stunned", "You asked me to marry you three weeks ago."),
            (D, "DANIEL", "Smug", "And you believed a man who needed your father's company."),
            (A, "Guests whisper behind champagne glasses. Mara's hand trembles on the banister."),
            (D, "MARA", "Furious", "You used me."),
            (D, "VICTORIA", "Mocking", "Everyone used you, darling. You were just the last to notice."),
            (D, "MARA", "Defiant", "Then let me be the first to leave."),
            (A, "Mara tears the diamond ring from her finger and hurls it into Daniel's champagne glass."),
            (D, "DANIEL", "Enraged", "You will regret this. No one in this city will ever hire you again!"),
            (D, "MARA", "Breathless", "Watch me."),
        ]),
        ("EXT. RAINY STREET \u2013 NIGHT", [
            (A, "Mara storms out into the downpour, mascara streaking down her cheeks."),
            (D, "STRANGER", "Low, Calm", "Miss Voss. Your father is alive."),
            (D, "MARA", "Stunned", "That is impossible. I buried him."),
            (D, "STRANGER", "Grave", "You buried an empty coffin. Get in the car before they find us."),
            (S, "Tires screech as a black sedan sweeps its headlights across the street."),
            (D, "MARA", "Desperate", "Who are you?"),
            (D, "STRANGER", "Urgent", "The man your father paid to keep you alive. Someone just put a price on your head."),
            (A, "A gunshot cracks. The sedan's rear window shatters beside Mara."),
            (D, "STRANGER", "Shouting", "Down!"),
        ]),
    ]),
    dict(n=2, logline="Mara hides in a safehouse and discovers the strangers' real employer is her ex-fiance's rival.",
         cont="Picks up directly from Episode 1's gunshot.", scenes=[
        ("INT. SAFEHOUSE \u2013 CONTINUOUS", [
            (A, "Mara crawls across the cracked floor of a dim safehouse, glass in her hair."),
            (D, "MARA", "Breathless", "Who is trying to kill me?"),
            (D, "STRANGER", "Cold", "Call me Kane. Whoever it is, they know exactly who you are."),
            (D, "MARA", "Furious", "Take me to my father. Now."),
            (D, "KANE", "Firm", "Not until you read this."),
            (A, "Kane slides a sealed letter across the table. Mara recognises her father's handwriting."),
            (D, "MARA", "Whispering", "He wrote my name wrong. He never does that."),
            (D, "KANE", "Grave", "It's a warning. Your father signed a contract with Daniel Ashford years ago."),
            (D, "MARA", "Stunned", "Daniel? My Daniel?"),
            (D, "KANE", "Low", "The wedding was never about love. It was collateral."),
            (A, "Mara's knees give way. She grips the table as thunder rattles the windows."),
            (D, "MARA", "Desperate", "Then why did he leave me alone tonight?"),
        ]),
        ("INT. SAFEHOUSE CORRIDOR \u2013 NIGHT", [
            (S, "A floorboard creaks beyond the door."),
            (D, "KANE", "Urgent", "Stay behind me. They found us faster than I expected."),
            (A, "The door handle turns slowly. A silhouette fills the frosted glass."),
            (D, "VICTORIA", "Smug", "Hello, cousin. Did you really think you could hide?"),
            (D, "MARA", "Stunned", "Victoria? You work for him?"),
            (D, "VICTORIA", "Mocking", "I never worked for him. He works for me."),
            (A, "Victoria lifts a pistol and aims it directly at Mara's chest."),
        ]),
    ]),
    dict(n=3, logline="Kane and Mara escape, but Mara learns Kane has been lying about who hired him.",
         cont="Follows Episode 2, where Victoria revealed herself as the true mastermind.", scenes=[
        ("EXT. ROOFTOP \u2013 DAWN", [
            (A, "Mara and Kane sprint across a rain-slicked rooftop as sirens wail below."),
            (D, "KANE", "Shouting", "Jump to the next building. Do not look down!"),
            (D, "MARA", "Terrified", "I can't!"),
            (D, "KANE", "Commanding", "You can. Your father jumped farther than this."),
            (A, "Mara leaps, lands hard, and rolls onto the gravel, gasping."),
            (D, "MARA", "Breathless", "You knew my father. Really knew him."),
            (D, "KANE", "Quiet", "I was his driver. Then his bodyguard. Then his friend."),
            (D, "MARA", "Cold", "Then why does his letter say never to trust you?"),
            (A, "Kane freezes. His hand drifts toward the holster at his hip."),
            (D, "KANE", "Low, Menacing", "Because he knew I would have to choose between you and my daughter."),
            (D, "MARA", "Stunned", "Your daughter?"),
            (D, "KANE", "Grave", "Victoria is my daughter, Mara. And she is not your cousin."),
            (S, "A deep, ominous string chord rises as the helicopter blades thunder overhead."),
        ]),
    ]),
]


def _timeline(ep):
    """Assign episode-global start/end times to every beat."""
    t, out = 0.0, []
    for heading, beats in ep["scenes"]:
        s0, items = t, []
        for b in beats:
            text = b[-1]
            dur = round(max(1.6, 0.42 * len(text.split()) + 1.0), 1) if b[0] == D else round(max(2.5, 0.2 * len(text.split()) + 1.5), 1)
            items.append((b, round(t, 1), round(t + dur, 1)))
            t = round(t + dur + (0.5 if b[0] == D else 0.0), 1)
        out.append((heading, s0, round(t, 1), items))
    return out


def render_kuku(ep) -> str:
    L = [f"EPISODE NUMBER: Episode {ep['n']}", f"EPISODE LOGLINE: {ep['logline']}", f"CONTINUITY NOTES: {ep['cont']}", "", "--- SCRIPT START ---", ""]
    for i, (heading, s0, s1, items) in enumerate(_timeline(ep), 1):
        L += [f"SCENE {i:02d}", f"Scene heading: {heading}", f"Scene start: {format_timecode(s0)}", f"Scene end: {format_timecode(s1)}", "", "Actions and dialogue:", ""]
        for b, a, z in items:
            if b[0] == D:
                L += [f"**{b[1]} ({b[2]}):**"]
            else:
                L += ["**ACTION:**" if b[0] == A else "**SOUND FX:**"]
            L += [f"Start: ({format_timecode(a)}) | End: ({format_timecode(z)})", f"\"{b[-1]}\"" if b[0] == D else b[-1], ""]
    L += ["--- SCRIPT END ---"]
    return "\n".join(L)


def render_variant(ep) -> str:
    """Messy variant: HH:MM:SS clocks, un-bolded labels, no scene end, a missing end time, an untimed line."""
    L = []
    if ep["n"] != 2:                       # episode 2 has NO header at all (number comes from filename)
        L += [f"episode number: {ep['n']}", f"logline: {ep['logline']}", ""]
    for i, (heading, s0, s1, items) in enumerate(_timeline(ep), 1):
        hh = lambda x: format_timecode(x + 3600 * 0)  # keep short; convert below
        def tc(x):
            h = int(x // 3600); m = int((x % 3600) // 60); s = x - h * 3600 - m * 60
            return f"{h:02d}:{m:02d}:{s:05.2f}"
        L += [f"Scene {i}", f"SCENE HEADING: {heading}", f"SCENE START: {tc(s0)}", ""]   # scene end intentionally omitted
        for k, (b, a, z) in enumerate(items):
            label = f"{b[1]} ({b[2]}):" if b[0] == D else ("ACTION:" if b[0] == A else "SOUND FX:")
            L += [label]
            if ep["n"] == 3 and i == 1 and k == 4:
                L += [f"Start: {tc(a)}", b[-1] if b[0] != D else f"\"{b[-1]}\"", ""]           # missing end time
            elif ep["n"] == 3 and i == 1 and k == 9:
                L += ["", b[-1] if b[0] != D else f"\"{b[-1]}\"", ""]                          # untimed line
            else:
                L += [f"Start: {tc(a)} - End: {tc(z)}", f"\u201c{b[-1]}\u201d" if b[0] == D else b[-1], ""]
    return "\r\n".join(L)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "sample_data"
    (root / "synthetic_show").mkdir(parents=True, exist_ok=True)
    (root / "synthetic_show_variant").mkdir(parents=True, exist_ok=True)
    for ep in EPISODES:
        (root / "synthetic_show" / f"episode-{ep['n']:03d}-Synthetic {ep['n']}.txt").write_text(render_kuku(ep), encoding="utf-8")
        name = f"ep{ep['n']:02d}_untitled.txt" if ep["n"] == 2 else f"variant-episode-{ep['n']}.txt"
        (root / "synthetic_show_variant" / name).write_bytes(("\ufeff" + render_variant(ep)).encode("utf-8"))
    print("wrote", root)


if __name__ == "__main__":
    main()
