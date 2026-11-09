"""Rule-based / lightweight-NLP signals about a time window of a screenplay.

These are used twice:
  1. by the candidate generator, to shortlist windows worth an LLM's attention, and
  2. by the RuleBasedModel, the free offline fallback "judge".

They combine lexical cues (reveal/threat/arrival words) with STRUCTURAL cues
(how the first 3 s and last seconds are built, turn-taking, escalation, scene
boundaries, number of speakers) so the score never rests on emotional keywords alone.
Every function returns (score 0-10, [human readable reasons]).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.models import Event

# --------------------------------------------------------------------------- lexicons
_I = re.I
ACCUSE_REVEAL = re.compile(
    r"\b(?:you (?:lied|betrayed|killed|knew|used|never|stole|poisoned|planned|did this|did it|are not|aren't|were never)"
    r"|it was you|it was (?:her|him|them)|i know (?:everything|what|who|about|the truth|you)|the truth|a lie|liar|traitor|"
    r"betray\w*|murder\w*|killed|poison\w*|pregnan\w*|bastard|illegitimate|secret\w*|forg(?:ed|ery)|conspir\w*|"
    r"blackmail\w*|confess\w*|expos(?:e|ed|ing)|never loved|not my (?:son|daughter|child|wife|husband)|"
    r"impost[eo]r|spy|treason|alive|dead|heir|fraud|swindl\w*|deceiv\w*|scheme\w*)\b", _I)
THREAT = re.compile(
    r"\b(?:i will (?:destroy|ruin|kill|make you|have you|see you|not let)|you will (?:pay|regret|die|hang|never)|or else|"
    r"how dare you|get out|seize (?:her|him|them)|guards?|arrest\w*|execut\w*|hang(?:ed)?|dungeon|banish\w*|disown\w*|"
    r"annul\w*|divorce|you'll regret|i'll (?:make|ruin|destroy|kill)|beg|mercy|kneel|crawl|last warning)\b", _I)
ARRIVAL = re.compile(
    r"\b(?:bursts?|barges?|storms? in|crashes? (?:in|through)|enters?|appears?|steps? (?:out|into|forward)|emerges?|"
    r"stranger|a figure|doors? (?:fl(?:ies|ing|ung)|swing\w*|burst\w*|slam\w*)|unexpected\w*|suddenly|arrives?|shadowy|"
    r"rushes? in|slips? in|looms?)\b", _I)
VIOLENCE = re.compile(
    r"\b(?:slaps?|slapp\w*|strikes?|struck|shoves?|pushes?|throws?|hurls?|draws? (?:a |his |her |the )?(?:sword|dagger|blade|pistol|knife)|"
    r"grabs?|seizes?|collapses?|stabs?|chokes?|drags?|smash\w*|slams?|punch\w*|kicks?|shatters?|snatch\w*|rips?|knocks?|"
    r"tears (?:the|her|his|it)|hammers?|pins?|restrain\w*)\b", _I)
SHOCK_REACT = re.compile(
    r"\b(?:gasps?|gasping|freezes?|frozen|eyes widen\w*|wide-eyed|stunned|horrified|horror|pales?|staggers?|trembl\w*|"
    r"speechless|disbelief|shock\w*|aghast|stumbles?|drops? (?:the|her|his)|stares? (?:in|at)|paralyz\w*)\b", _I)
STING = re.compile(
    r"\b(?:sting|stinger|ominous|cliffhanger|dramatic (?:chord|boom|swell|sting)|sharp (?:gasp|crack|intake)|crescendo|"
    r"thunder\w*|boom\w*|crash\w*|scream\w*|gunshot|dissonant|suspense\w*|tension|foreboding|menacing)\b", _I)
RESOLUTION = re.compile(
    r"\b(?:thank you|thanks|forgive me|i forgive|farewell|goodbye|good night|good evening|welcome home|i love you too|"
    r"at last|finally|it is settled|it is done|we are safe|all is well|peace|you are safe|rest now|"
    r"everything will be (?:fine|alright|okay)|my (?:dear|love)|i promise you|i am here)\b", _I)
CLOSURE_ACTION = re.compile(
    r"\b(?:smiles? (?:warmly|softly|gently|tenderly|brightly)|embrac\w*|kiss\w*|relieved|relief|laughs?|"
    r"walks? (?:away|off|together)|silhouettes? disappear\w*|hand in hand|side by side|peaceful\w*|tender\w*|"
    r"cups? .{0,20}cheek|nods? (?:slowly|gently|softly)|bows? (?:her|his) head in)\b", _I)
GREETING = re.compile(r"^\W*(?:good (?:morning|evening|afternoon|day)|hello|hi\b|welcome|please (?:sit|come in)|"
                      r"thank you|how are you|my lady,? (?:i hope|how are))", _I)
CONNECTIVE_START = re.compile(r"^\W*(?:\.\.\.|\u2026|and\b|but\b|so\b|then\b|because\b|which\b|or\b|nor\b|yet\b|that is why\b)", _I)
REACTIVE_OPENER = re.compile(
    r"^\W*(?:yes|no|yeah|nay|fine|very well|i see|perhaps|maybe|indeed|okay|alright|nothing|what do you mean|what wedding|"
    r"what|why|who|how|when|where|really|and|but|you knew|you did|so|then|oh|ah|well)\W*$", _I)
PRONOUN_START = re.compile(r"^\W*(?:he|she|they|it|that|this|those|these|him|her|them)\b", _I)
BACKREF = re.compile(r"\b(?:as i (?:said|told)|like i said|last time|earlier|yesterday|remember when|what happened (?:the other|last)|"
                     r"i told you|as before|once again|the other day)\b", _I)
RELATION = re.compile(
    r"\b(?:wife|husband|fianc\w*|sister|brother|mother|father|daughter|son|duke|duchess|lady|lord|king|queen|prince|princess|"
    r"judge|soldier|general|marriage|married|wedding|bride|groom|heir|servant|maid|steward|priest|countess|baron|cousin)\b", _I)
INTENSE_DELIVERY = re.compile(
    r"(?:desperate|furious|enraged|screaming|shouting|trembl\w*|weeping|sobbing|pleading|panick\w*|horrified|shaken|"
    r"menacing|venom\w*|cold|cruel|mocking|defiant|breathless|urgent|gloating|seething|broken|anguish\w*|stricken|"
    r"incredulous|unbelieving|commanding|demanding|fierce|icy|bitter|smug|hysterical|terrified|stunned)", _I)
EMO_ACTION = re.compile(
    r"\b(?:tears?|sobs?|weeps?|cries|crying|trembl\w*|clutch\w*|collapses?|gasps?|scream\w*|flinch\w*|recoil\w*|"
    r"clench\w*|whiteknuckle\w*|white-knuckled|shak(?:es|ing)|devastat\w*|heartbr\w*|fury|rage|defian\w*)\b", _I)
EMO_WORDS = re.compile(
    r"\b(?:betray\w*|lie|lies|lied|kill\w*|die|dead|death|love|hate|never|forever|please|beg|mother|child|blood|"
    r"shame|honou?r|trust|destroy\w*|ruin\w*|alone|afraid|fear|pain|heart|sorry|mercy)\b", _I)

TAG_RULES: List[Tuple[str, re.Pattern]] = [
    ("betrayal", re.compile(r"\b(?:betray\w*|traitor|lied to|deceiv\w*|behind my back|backstab\w*|cheat\w*|abandon\w*|disown\w*)\b", _I)),
    ("confrontation", re.compile(r"\b(?:how dare|you will answer|face me|explain yourself|enough|accus\w*|admit|answer me|you (?:will|must) (?:tell|explain))\b", _I)),
    ("shocking_revelation", re.compile(r"\b(?:the truth|pregnan\w*|never told|you are not|you're not|it was you|real identity|heir|alive|bastard|illegitimate|all along|all this time)\b", _I)),
    ("secret_exposed", re.compile(r"\b(?:secret\w*|expos\w*|hidden|evidence|letter|proof|forg(?:ed|ery)|reveal\w*|confess\w*|ledger)\b", _I)),
    ("unexpected_arrival", ARRIVAL),
    ("threat", THREAT),
    ("romantic_conflict", re.compile(r"\b(?:love|kiss\w*|heart|marry|marriage|wedding|fianc\w*|husband|wife|mistress|jealous\w*|bride|groom)\b", _I)),
    ("emotional_breakdown", re.compile(r"\b(?:tears?|sobs?|weep\w*|collapses?|breaks? down|trembl\w*|devastat\w*|heartbr\w*)\b", _I)),
]

CONTENT = ("dialogue", "action", "sfx")


# --------------------------------------------------------------------------- window view
@dataclass
class Window:
    start: float
    end: float
    events: List[Event]                      # content events (dialogue/action/sfx) inside, in time order
    episode_end: float = 0.0
    scene_bounds: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def dialogue(self) -> List[Event]:
        return [e for e in self.events if e.kind == "dialogue"]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def text(self) -> str:
        return " ".join(e.text for e in self.events if e.kind != "sfx")


def build_window(events: Sequence[Event], start: float, end: float, episode_end: float = 0.0,
                 scene_bounds: Optional[List[Tuple[float, float]]] = None) -> Window:
    inside = [e for e in events if e.kind in CONTENT and e.start is not None and e.end is not None
              and e.start >= start - 0.05 and e.start < end - 0.001]
    inside.sort(key=lambda e: (e.start, e.end))
    return Window(start, end, inside, episode_end, scene_bounds or [])


def window_is_clean(events: Sequence[Event], start: float, end: float) -> bool:
    """A window may not slice through a spoken line (dialogue must be fully inside)."""
    for e in events:
        if e.kind != "dialogue" or e.start is None or e.end is None:
            continue
        starts_before = e.start < start - 0.05 < e.end - 0.05
        ends_after = e.start < end - 0.05 and e.end > end + 0.05
        if starts_before or ends_after:
            return False
    return True


def _clamp(x: float) -> float:
    return max(0.0, min(10.0, round(x, 2)))


# --------------------------------------------------------------------------- signals
def hook_signal(w: Window, hook_seconds: float = 3.0) -> Tuple[float, List[str]]:
    zone = [e for e in w.events if e.kind in ("dialogue", "action") and e.start < w.start + hook_seconds]
    if not zone:
        return 0.0, ["nothing is said or happens in the first 3 s"]
    d = [e for e in zone if e.kind == "dialogue"]
    a = [e for e in zone if e.kind == "action"]
    score, why = 2.0, []
    if d:
        first = d[0]
        dtxt = " ".join(e.text for e in d)
        if CONNECTIVE_START.match(first.text):
            score -= 2.5; why.append("opens mid-sentence (continuation line)")
        if GREETING.match(first.text):
            score -= 2.0; why.append("opens with a pleasantry")
        if REACTIVE_OPENER.match(first.text):
            score -= 3.0; why.append("opens on a bare reaction to something the viewer has not seen")
        if "?" in dtxt:
            score += 2.0; why.append("opens with a question")
        if "!" in dtxt:
            score += 1.5; why.append("opens with an exclamation")
        lex = 0.0
        if ACCUSE_REVEAL.search(dtxt):
            lex += 2.0; why.append("accusation/revelation wording in first 3 s")
        if THREAT.search(dtxt):
            lex += 2.0; why.append("threat/command wording in first 3 s")
        score += min(lex, 3.5)
        n = len(first.text.split())
        if n <= 8:
            score += 1.0; why.append("short punchy opening line")
        elif n > 25:
            score -= 1.5; why.append("long expository opening line")
        if INTENSE_DELIVERY.search(first.delivery):
            score += 1.0; why.append(f"intense delivery ({first.delivery})")
    atxt = " ".join(e.text for e in a)
    if a and (VIOLENCE.search(atxt) or ARRIVAL.search(atxt) or SHOCK_REACT.search(atxt)):
        score += 2.0; why.append("physical/arrival beat in first 3 s")
    if not d and not (a and VIOLENCE.search(atxt)):
        score -= 1.5; why.append("no dialogue and no striking action in first 3 s")
    return _clamp(score), why


def cliffhanger_signal(w: Window) -> Tuple[float, List[str]]:
    dlg = w.dialogue
    if not w.events:
        return 0.0, ["empty window"]
    score, why = 2.0, []
    tail_d = dlg[-2:]
    tail_txt = " ".join(e.text for e in tail_d)
    last_d = dlg[-1] if dlg else None
    last_ev = max((e for e in w.events if e.kind in ("dialogue", "action")), key=lambda e: e.end, default=None)
    if last_d is not None:
        t = last_d.text.rstrip()
        if t.endswith("?"):
            score += 2.5; why.append("ends on an unanswered question")
        elif t.endswith(("...", "\u2026", "\u2014", "--")):
            score += 1.5; why.append("last line trails off / is cut short")
        if ACCUSE_REVEAL.search(tail_txt) or THREAT.search(tail_txt):
            score += 2.5; why.append("final lines contain a reveal or threat")
        if RESOLUTION.search(tail_txt):
            score -= 3.0; why.append("final lines sound resolved (thanks/farewell/reassurance)")
        if len(last_d.text.split()) >= 25:
            score -= 1.0; why.append("ends on a long wrap-up speech")
        if "!" in last_d.text:
            score += 0.5
    tail_events = [e for e in w.events if e.end >= w.end - 3.0]
    tail_act = " ".join(e.text for e in tail_events if e.kind == "action")
    tail_sfx = " ".join(e.text for e in tail_events if e.kind == "sfx")
    if tail_act and ARRIVAL.search(tail_act):
        score += 2.0; why.append("ends on an arrival/interruption")
    if tail_act and SHOCK_REACT.search(tail_act):
        score += 1.5; why.append("ends on a shocked reaction")
    if tail_sfx and STING.search(tail_sfx):
        score += 1.0; why.append("dramatic sound sting at the cut")
    if tail_act and CLOSURE_ACTION.search(tail_act):
        score -= 2.0; why.append("ends on a calm/closing gesture")
    if last_ev is not None and w.episode_end and w.end >= w.episode_end - 1.5:
        score += 0.5; why.append("window runs to the episode's own end")
    return _clamp(score), why


def standalone_signal(w: Window) -> Tuple[float, List[str]]:
    dlg = w.dialogue
    if not dlg:
        return 1.0, ["no dialogue to follow"]
    score, why = 5.0, []
    speakers = {e.speaker for e in dlg if e.speaker}
    if len(speakers) >= 2:
        score += 1.5; why.append(f"{len(speakers)} speakers interact")
    else:
        score -= 2.0; why.append("a single speaker (no visible conflict)")
    if len(speakers) > 4:
        score -= min(3.0, len(speakers) - 4); why.append(f"{len(speakers)} speakers is hard to follow cold")
    if any(abs(w.start - s) <= 2.0 for s, _ in w.scene_bounds):
        score += 1.0; why.append("starts at a scene start (setting is established)")
    if PRONOUN_START.match(dlg[0].text):
        score -= 1.0; why.append("first line refers to someone by pronoun only")
    rel = {m.group(0).lower() for m in RELATION.finditer(w.text())}
    if rel:
        score += 1.0; why.append("relationships/roles are named")
    back = BACKREF.findall(w.text())
    if back:
        score -= min(2.0, len(back)); why.append("refers back to events outside the clip")
    if w.dialogue and len(w.dialogue) >= 4:
        score += 0.5
    return _clamp(score), why


def emotion_signal(w: Window) -> Tuple[float, List[str]]:
    dlg = w.dialogue
    score, why = 2.0, []
    if not dlg:
        return 1.0, ["no dialogue"]
    intense = sum(1 for e in dlg if INTENSE_DELIVERY.search(e.delivery))
    if intense:
        score += min(3.0, 0.6 * intense); why.append(f"{intense} intensely-directed lines")
    excl = sum(e.text.count("!") for e in dlg)
    if excl:
        score += min(1.5, 0.4 * excl)
    q = sum(e.text.count("?") for e in dlg)
    if q:
        score += min(1.0, 0.3 * q)
    act_txt = " ".join(e.text for e in w.events if e.kind == "action")
    emo_act = len(EMO_ACTION.findall(act_txt))
    if emo_act:
        score += min(2.0, 0.5 * emo_act); why.append("physical emotional beats (tears, clutching, gasps...)")
    emo_words = len(EMO_WORDS.findall(" ".join(e.text for e in dlg)))
    if emo_words:
        score += min(1.5, 0.3 * emo_words)
    if w.duration > 0 and len(dlg) / (w.duration / 10) >= 2.0:
        score += 1.0; why.append("fast back-and-forth exchange")
    # escalation: is the second half hotter than the first? (structural, not lexical)
    mid = w.start + w.duration / 2
    def heat(evs):
        return sum(("!" in e.text) + ("?" in e.text) + bool(INTENSE_DELIVERY.search(e.delivery)) for e in evs)
    first, second = [e for e in dlg if e.start < mid], [e for e in dlg if e.start >= mid]
    if second and heat(second) / len(second) > heat(first) / max(1, len(first)) + 0.2:
        score += 1.0; why.append("intensity escalates toward the end")
    return _clamp(score), why


def moment_tags(w: Window) -> List[str]:
    txt = w.text()
    tags = [name for name, rx in TAG_RULES if rx.search(txt)]
    if w.dialogue and w.dialogue[-1].text.rstrip().endswith("?"):
        tags.append("suspenseful_question")
    # keep only the most informative few, preserving order
    return tags[:6]


def all_signals(w: Window, hook_seconds: float = 3.0) -> Dict[str, Tuple[float, List[str]]]:
    return {"hook": hook_signal(w, hook_seconds), "cliffhanger": cliffhanger_signal(w),
            "standalone": standalone_signal(w), "emotion": emotion_signal(w)}
